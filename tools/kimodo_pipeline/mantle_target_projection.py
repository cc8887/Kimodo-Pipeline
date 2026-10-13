"""Bounded source-pose conditioning informed by the unchanged ALS target budget.

These poses are generation conditions, never repaired delivery assets. The
observer shares the angular converter's mapping, neutral alignment and scale;
every regenerated clip still requires source, target, FBX and runtime gates.
"""
from __future__ import annotations

import copy
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from .gait_constraints import _rotation_between
from .motion import validate_motion
from .offline_retarget import BASIS, aligned_neutral, mapping
from .prompt_modules import fingerprint

POLICY = 'mantle-source-target-feasibility-v1'


class TargetProjectionError(ValueError):
    def __init__(self, message, receipt):
        self.receipt = receipt
        super().__init__(message)


def _fk(local, root, rig):
    rotations = np.empty_like(local)
    positions = np.empty((len(local), 3))
    for index, parent in enumerate(rig['parents']):
        if parent < 0:
            rotations[index], positions[index] = local[index], root
        else:
            rotations[index] = rotations[parent] @ local[index]
            positions[index] = positions[parent] + rotations[parent] @ rig['offsets'][index]
    return rotations, positions


class TargetPoseObserver:
    """Cached pre-cleanup angular-retarget FK, in ALS centimeters/Z-up."""
    def __init__(self, rig, skeleton):
        self.rig = copy.deepcopy(rig)
        self.skeleton = copy.deepcopy(skeleton)
        self.identity = fingerprint(dict(rig=rig, skeleton=skeleton))
        self.names = [name.lower() for name in skeleton['rawBoneNames']]
        self.lookup = {name: index for index, name in enumerate(self.names)}
        self.parents = skeleton['rawParents']
        source_names = {name: index for index, name in enumerate(rig['jointNames'])}
        if (len(self.lookup) != len(self.names) or len(self.parents) != len(self.names) or
                any(parent < -1 or parent >= index for index, parent in enumerate(self.parents))):
            raise ValueError('Target observer requires unique parent-first physical bones')
        local_scales = np.asarray([item['scale'] for item in skeleton['referencePose'][:len(self.names)]], dtype=float)
        if local_scales.shape != (len(self.names), 3) or not np.isfinite(local_scales).all() or np.any(local_scales <= 0):
            raise ValueError('Target observer requires finite positive reference scales')
        self.world_scales = local_scales.copy()
        for index, parent in enumerate(self.parents):
            if parent >= 0:
                self.world_scales[index] *= self.world_scales[parent]
        # Converter twist reconstruction can change only auxiliary leaves.
        for index, name in enumerate(self.names):
            if '_twist_' in name and index in self.parents:
                raise ValueError('Twist observer requires auxiliary leaf bones')
        bone_map = mapping()
        self.mapped = {index: source_names[bone_map[name]] for index, name in enumerate(self.names)
                       if name in bone_map}
        self.aligned = aligned_neutral(skeleton, rig, bone_map)
        self.offsets = np.asarray([item['position'] for item in skeleton['referencePose'][:len(self.names)]], dtype=float)
        source_leg = sum(np.linalg.norm(rig['offsets'][source_names['Left'+part]]) for part in ('Shin', 'Foot'))
        target_leg = sum(np.linalg.norm(self.offsets[self.lookup[part]]) for part in ('calf_l', 'foot_l'))
        if not np.isfinite(source_leg + target_leg) or min(source_leg, target_leg) <= 0:
            raise ValueError('Invalid retarget limb scale')
        self.scale = target_leg / (100. * source_leg)

    def evaluate(self, local, root):
        source_rotations, _ = _fk(local, root, self.rig)
        rotations = np.empty((len(self.names), 3, 3))
        points = np.empty((len(self.names), 3))
        for index, (name, parent) in enumerate(zip(self.names, self.parents)):
            if index in self.mapped:
                rotations[index] = BASIS @ source_rotations[self.mapped[index]] @ BASIS.T @ self.aligned[index]
            elif parent < 0:
                rotations[index] = self.aligned[index]
            else:
                rotations[index] = rotations[parent] @ self.aligned[parent].T @ self.aligned[index]
            offset = BASIS @ root * (100. * self.scale) if name == 'pelvis' else self.offsets[index]
            points[index] = offset if parent < 0 else points[parent] + rotations[parent] @ (self.world_scales[parent]*offset)
        return rotations, points

    observe = evaluate

    def compare_convert(self, local, root, *, bones=('hand_l', 'hand_r')):
        """Independent production-converter parity check, including leaf twists."""
        from .motion import complete
        from .offline_retarget import convert
        from .publish import world_pose
        timing = dict(sampledKeyCount=1, playLength=0., frameRateNumerator=30, frameRateDenominator=1)
        raw, _ = convert(complete(local[None], root[None], self.rig), self.rig,
                         self.skeleton, 'conditioning-observer-parity', timing)
        actual = world_pose(raw, self.skeleton)
        predicted_rot, predicted_pos = self.evaluate(local, root)
        rows = []
        for bone in bones:
            index = self.lookup[bone]
            rows.append(dict(bone=bone, positionErrorCm=float(np.linalg.norm(predicted_pos[index]-actual[bone][1][0])),
                orientationErrorDegrees=float(np.rad2deg(Rotation.from_matrix(predicted_rot[index].T @ actual[bone][0][0]).magnitude()))))
        return dict(passed=all(row['positionErrorCm'] < 1e-6 and row['orientationErrorDegrees'] < 1e-6 for row in rows), bones=rows)


def _contacts(items, label):
    result = {}
    for item in items:
        side = item['side']
        if side not in ('left', 'right') or item.get('kind', 'hand') != 'hand' or side in result:
            raise ValueError(label + ' requires distinct left/right hands')
        position, rotation = np.asarray(item['position'], dtype=float), np.asarray(item['rotation'], dtype=float)
        if (position.shape != (3,) or rotation.shape != (3, 3) or not np.isfinite(position).all() or
                not np.isfinite(rotation).all() or not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-7) or
                not np.isclose(np.linalg.det(rotation), 1., atol=1e-7)):
            raise ValueError('Invalid ' + label + ' contact transform')
        result[side] = (position, rotation)
    return result


def project_target_pose(local, root, rig, source_contacts, reference_root, target_skeleton, target_contacts, *,
                        budget_cm=10., margin_cm=.5, joint_limit_degrees=30., pole_limit_degrees=60.,
                        maximum_root_correction_meters=.03, observer=None):
    """Find a small source change with exact source wrists and feasible target FK.

    Root remains exactly unchanged whenever the first, fixed-root solve works.
    Torso/clavicle deltas and elbow-pole turns are bounded; free local joints
    and all bind offsets remain unchanged. Infeasible requests raise with a
    failed receipt rather than enlarging the unchanged target repair budget.
    """
    local, root, reference_root = (np.asarray(value, dtype=float) for value in (local, root, reference_root))
    validate_motion(dict(local_rot_mats=local[None], root_positions=root[None], fps=30), rig)
    values = (budget_cm, margin_cm, joint_limit_degrees, pole_limit_degrees, maximum_root_correction_meters)
    if (not all(np.isfinite(values)) or not 0 < budget_cm <= 10 or not 0 <= margin_cm < budget_cm or
            not 0 < joint_limit_degrees <= 40 or not 0 <= pole_limit_degrees <= 90 or
            not 0 <= maximum_root_correction_meters <= .15 or reference_root.shape != (3,) or
            not np.isfinite(reference_root).all()):
        raise ValueError('Invalid target/source projection budget')
    sources, targets = _contacts(source_contacts, 'source'), _contacts(target_contacts, 'target')
    if not sources or set(sources) != set(targets):
        raise ValueError('Source and target must name the same support hands')
    observer = observer or TargetPoseObserver(rig, target_skeleton)
    if observer.identity != fingerprint(dict(rig=rig, skeleton=target_skeleton)):
        raise ValueError('Observer belongs to different rig or target skeleton')
    names = {name: index for index, name in enumerate(rig['jointNames'])}
    sides = sorted(sources)
    adjustable = [names[name] for name in ['Hips', 'Spine1', 'Spine2', 'Chest',
                  *[side.title()+'Shoulder' for side in sides]] if name in names]
    limbs = []
    for side in sides:
        indices = [names[side.title()+part] for part in ('Arm', 'ForeArm', 'Hand')]
        if rig['parents'][indices[1]] != indices[0] or rig['parents'][indices[2]] != indices[1]:
            raise ValueError('Source support requires a two-link arm')
        lengths = [float(np.linalg.norm(rig['offsets'][index])) for index in indices[1:]]
        if min(lengths) <= 0:
            raise ValueError('Source arm has zero length')
        limbs.append((indices, lengths))
    angular_limit, pole_limit = np.deg2rad([joint_limit_degrees, pole_limit_degrees])
    target_limit = budget_cm-margin_cm
    _, base_target_pos = observer.evaluate(local, root)
    dimensions = 3*len(adjustable)+len(sides)

    def evaluate(parameters, move_root):
        result = local.copy()
        for index, delta in zip(adjustable, parameters[:3*len(adjustable)].reshape(-1, 3)):
            result[index] = local[index] @ Rotation.from_rotvec(delta).as_matrix()
        position = root+parameters[dimensions:] if move_root else root.copy()
        reach = []
        for number, (side, (indices, lengths)) in enumerate(zip(sides, limbs)):
            rotations, points = _fk(result, position, rig)
            arm, forearm, hand = indices
            anchor, hand_rotation = sources[side]
            direction = anchor-points[arm]
            distance = float(np.linalg.norm(direction))
            lower, upper = abs(lengths[0]-lengths[1])+1e-6, sum(lengths)-1e-6
            reach.extend([distance-lower, upper-distance])
            direction = direction/max(distance, 1e-12)
            if distance < 1e-12:
                direction = np.array([1., 0., 0.])
            pole = points[forearm]-points[arm]
            pole -= direction*float(pole @ direction)
            if np.linalg.norm(pole) < 1e-10:
                axis = np.eye(3)[int(np.argmin(np.abs(direction)))]
                pole = axis-direction*float(axis @ direction)
            pole = Rotation.from_rotvec(direction*parameters[3*len(adjustable)+number]).apply(pole/np.linalg.norm(pole))
            # Trial poses may be unreachable while SLSQP searches. Clamp only
            # their temporary IK construction; explicit reach inequalities and
            # independent final wrist reconstruction reject such solutions.
            fitted_distance = float(np.clip(distance, lower, upper))
            along = (lengths[0]**2-lengths[1]**2+fitted_distance**2)/(2*fitted_distance)
            elbow = points[arm]+direction*along+pole*np.sqrt(max(0., lengths[0]**2-along**2))
            endpoint = points[arm]+direction*fitted_distance
            arm_global = _rotation_between(points[forearm]-points[arm], elbow-points[arm]) @ rotations[arm]
            forearm_global = _rotation_between(points[hand]-points[forearm], endpoint-elbow) @ rotations[forearm]
            parent = rig['parents'][arm]
            result[arm] = rotations[parent].T @ arm_global if parent >= 0 else arm_global
            result[forearm] = arm_global.T @ forearm_global
            result[hand] = forearm_global.T @ hand_rotation
        target_rot, target_pos = observer.evaluate(result, position)
        inequalities = [(.15**2-float(np.sum((position-reference_root)**2)))/.15**2, *reach]
        for delta in parameters[:3*len(adjustable)].reshape(-1, 3):
            inequalities.append(1.-float(delta @ delta)/angular_limit**2)
        for side in sides:
            hand = observer.lookup['hand_'+side[0]]
            inequalities.append(1.-float(np.sum((target_pos[hand]-targets[side][0])**2))/target_limit**2)
        if move_root:
            inequalities.append(1.-float(np.sum(parameters[dimensions:]**2))/maximum_root_correction_meters**2)
        return result, position, np.asarray(inequalities)

    def objective(parameters):
        torso = parameters[:3*len(adjustable)] / np.deg2rad(10.)
        poles = parameters[3*len(adjustable):dimensions] / np.deg2rad(30.)
        return float(torso @ torso + poles @ poles + np.sum((parameters[dimensions:]/.01)**2))

    attempts = []
    solution = None
    for move_root in (False, True):
        if move_root and maximum_root_correction_meters == 0:
            continue
        initial = np.zeros(dimensions+(3 if move_root else 0))
        bounds = [(-angular_limit, angular_limit)]*(3*len(adjustable))+[(-pole_limit, pole_limit)]*len(sides)
        if move_root:
            bounds += [(-maximum_root_correction_meters, maximum_root_correction_meters)]*3
        cached_parameters, cached_value = None, None
        def cached(parameters):
            nonlocal cached_parameters, cached_value
            if cached_parameters is None or not np.array_equal(parameters, cached_parameters):
                cached_parameters, cached_value = parameters.copy(), evaluate(parameters, move_root)
            return cached_value
        fitted = minimize(objective, initial, method='SLSQP', bounds=bounds,
            constraints=[dict(type='ineq', fun=lambda parameters: cached(parameters)[2])],
            options=dict(maxiter=220, ftol=1e-10))
        result, position, inequalities = cached(fitted.x)
        attempts.append(dict(rootAllowed=move_root, solverSuccess=bool(fitted.success),
            solverMessage=str(fitted.message), iterations=int(fitted.nit), minimumConstraintValue=float(min(inequalities))))
        if fitted.success and min(inequalities) >= -1e-7:
            solution = (result, position, fitted.x)
            break
    receipt = dict(schemaVersion=1, policy=POLICY, scope='generation_conditions_only', deliverable=False,
        productionReady=False, passed=False, observerIdentity=observer.identity, targetRepairBudgetCm=budget_cm,
        conditioningMarginCm=margin_cm, rootGateMeters=.15, maximumRootCorrectionMeters=maximum_root_correction_meters,
        maximumTorsoJointDeltaDegrees=joint_limit_degrees, maximumElbowPoleDeltaDegrees=pole_limit_degrees, attempts=attempts,
        originalTargetCorrectionsCm={side:float(np.linalg.norm(base_target_pos[observer.lookup['hand_'+side[0]]]-targets[side][0]))
                                     for side in sides})
    if solution is None:
        raise TargetProjectionError('No feasible bounded source/target conditioning pose', receipt)
    result, position, parameters = solution
    rotations, points = _fk(result, position, rig)
    target_rotations, target_points = observer.evaluate(result, position)
    changes = np.rad2deg(Rotation.from_matrix(local.swapaxes(-1, -2) @ result).magnitude())
    allowed = set(adjustable) | {index for indices, _ in limbs for index in indices}
    free_unchanged = all(np.array_equal(local[index], result[index]) for index in range(len(local)) if index not in allowed)
    bone_error = max((abs(float(np.linalg.norm(points[index]-points[parent]))-float(np.linalg.norm(rig['offsets'][index])))
                      for index, parent in enumerate(rig['parents']) if parent >= 0), default=0.)
    receipt.update(rootCorrectionCm=float(np.linalg.norm(position-root)*100),
        rootErrorAfterCm=float(np.linalg.norm(position-reference_root)*100), freeLocalJointsPreserved=free_unchanged,
        maximumBoneLengthErrorCm=bone_error*100,
        referenceFullBodyPoseCopied=False, affectedJoints=[dict(name=name, localRotationChangeDegrees=float(changes[index]))
            for name, index in names.items() if changes[index] > 1e-7],
        elbowPoleDeltasDegrees={side:float(np.rad2deg(parameters[3*len(adjustable)+index])) for index, side in enumerate(sides)},
        contacts=[])
    for side in sides:
        source_hand, target_hand = names[side.title()+'Hand'], observer.lookup['hand_'+side[0]]
        receipt['contacts'].append(dict(side=side,
            sourcePositionErrorCm=float(np.linalg.norm(points[source_hand]-sources[side][0])*100),
            sourceOrientationErrorDegrees=float(np.rad2deg(Rotation.from_matrix(sources[side][1].T @ rotations[source_hand]).magnitude())),
            originalTargetCorrectionCm=float(np.linalg.norm(base_target_pos[target_hand]-targets[side][0])),
            projectedTargetCorrectionCm=float(np.linalg.norm(target_points[target_hand]-targets[side][0])),
            targetOrientationErrorDegrees=float(np.rad2deg(Rotation.from_matrix(targets[side][1].T @ target_rotations[target_hand]).magnitude()))))
    receipt['passed'] = (free_unchanged and bone_error < 1e-8 and receipt['rootErrorAfterCm'] <= 15.+1e-5 and
        all(row['sourcePositionErrorCm'] < 1e-4 and row['sourceOrientationErrorDegrees'] < 1e-4 and
            row['projectedTargetCorrectionCm'] <= target_limit+1e-5 and row['targetOrientationErrorDegrees'] <= 30.+1e-5
            for row in receipt['contacts']))
    if not receipt['passed']:
        raise TargetProjectionError('Independent source/target reconstruction failed', receipt)
    return result, position, receipt

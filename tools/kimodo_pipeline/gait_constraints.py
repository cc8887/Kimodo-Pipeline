"""Build sparse Kimodo gait conditions from measured reference events.

A discovery clip can supply conditions without becoming an accepted master.
Only constraint consistency is certified here. The generated result must pass
the independent authoring, target-skeleton and runtime gates.
"""
from __future__ import annotations

import hashlib
import json
import math

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from .gait import audit_candidate
from .motion import complete, fk, validate_motion


class GaitConstraintError(ValueError):
    def __init__(self, message, receipt):
        self.receipt = receipt
        super().__init__(message)


PELVIS_BACKEND = 'kimodo-python-named-feet-then-hips-v1'


def validate_pelvis_constraint_consistency(constraints, rig, smooth_path, pelvis_residual, headings, *, maximum_frames=19):
    """Validate equal-frame geometry before the backend's ordered extraction.

    Named foot sets retain Kimodo postprocess masks. A synchronized Hips EE
    follows them; fullbody is ordered last by the supported backend. Ordering
    cannot excuse conflicting values at any shared frame.
    """
    smooth_path, pelvis_residual, headings = map(np.asarray, (smooth_path, pelvis_residual, headings))
    count = len(smooth_path)
    if smooth_path.shape != (count, 2) or pelvis_residual.shape != (count, 2) or headings.shape != (count,):
        raise ValueError('Invalid expected pelvis/smooth-root constraint domain')
    if not all(np.isfinite(value).all() for value in (smooth_path, pelvis_residual, headings)):
        raise ValueError('Non-finite expected pelvis/smooth-root constraint domain')
    if not constraints or constraints[0].get('type') != 'root2d':
        raise ValueError('Preserved pelvis constraints require root2d first')
    root2d = constraints[0]
    if root2d.get('frame_indices') != list(range(count)) or not np.allclose(
            root2d.get('smooth_root_2d'), smooth_path, atol=1e-8, rtol=0):
        raise ValueError('Smooth-root constraints differ from the requested trajectory')
    expected_heading = np.column_stack((np.cos(headings), np.sin(headings)))
    if not np.allclose(root2d.get('global_root_heading'), expected_heading, atol=1e-7, rtol=0):
        raise ValueError('Smooth-root heading differs from the requested heading')
    frames_by_type, at_frame, foot_orders, hips_orders = {}, {}, [], []
    maximum_shared_position_error, maximum_shared_rotation_error = 0., 0.
    names = {name: index for index, name in enumerate(rig['jointNames'])}
    hips = rig.get('hipJointNames', ['RightLeg', 'LeftLeg'])
    for order, item in enumerate(constraints[1:], 1):
        kind, frames = item.get('type'), item.get('frame_indices')
        if kind not in ('fullbody', 'left-foot', 'right-foot', 'end-effector'):
            raise ValueError('Unsupported type in preserved pelvis constraints')
        if not isinstance(frames, list) or not frames or any(type(frame) is not int or not 0 <= frame < count for frame in frames) or len(frames) != len(set(frames)):
            raise ValueError('Invalid sparse pelvis frame indices')
        frames_by_type.setdefault(kind, set()).update(frames)
        if len(frames_by_type[kind]) > maximum_frames:
            raise ValueError('Preserved pelvis sparse constraint frame budget exceeded')
        if kind == 'end-effector':
            if item.get('joint_names') != ['Hips']:
                raise ValueError('Preserved pelvis adapter requires synchronized Hips EE')
            hips_orders.append(order)
        elif kind.endswith('-foot'):
            foot_orders.append(order)
        if not np.allclose(item.get('smooth_root_2d'), smooth_path[frames], atol=1e-8, rtol=0):
            raise ValueError('Sparse smooth-root constraints contradict root2d')
        axis = np.asarray(item.get('local_joints_rot'), dtype=float)
        root = np.asarray(item.get('root_positions'), dtype=float)
        if axis.shape != (len(frames), len(names), 3) or root.shape != (len(frames), 3) or not np.isfinite(axis).all() or not np.isfinite(root).all():
            raise ValueError('Invalid sparse pelvis pose payload')
        if not np.allclose(root[:, [0, 2]]-smooth_path[frames], pelvis_residual[frames], atol=1e-8, rtol=0):
            raise ValueError('Sparse Hips residual differs from verified native pelvis path')
        local = Rotation.from_rotvec(axis.reshape(-1, 3)).as_matrix().reshape(len(frames), len(names), 3, 3)
        _global, points = fk(local, root, rig)
        line = points[:, names[hips[0]]] - points[:, names[hips[1]]]
        observed = np.arctan2(line[:, 2], -line[:, 0])
        difference = observed-headings[frames]
        if np.max(np.abs(np.arctan2(np.sin(difference), np.cos(difference)))) > 1e-7:
            raise ValueError('Sparse pelvis pose contradicts root2d heading')
        for index, frame in enumerate(frames):
            if frame in at_frame:
                old_local, old_points = at_frame[frame]
                position_error = float(np.linalg.norm(points[index]-old_points, axis=-1).max())
                rotation_error = float(np.linalg.norm(Rotation.from_matrix(local[index]@old_local.swapaxes(-1,-2)).as_rotvec(),axis=-1).max())
                maximum_shared_position_error = max(maximum_shared_position_error, position_error)
                maximum_shared_rotation_error = max(maximum_shared_rotation_error, rotation_error)
                if position_error > 1e-8 or rotation_error > 1e-7:
                    raise ValueError('Conflicting sparse constraint poses at the same frame')
            at_frame[frame] = local[index], points[index]
    if (not hips_orders or not foot_orders or min(hips_orders) <= max(foot_orders) or
            not {'left-foot', 'right-foot'} <= set(frames_by_type)):
        raise ValueError('Backend requires named feet followed by synchronized Hips EE')
    needed = set().union(*(frames for kind, frames in frames_by_type.items() if kind != 'end-effector'))
    if frames_by_type['end-effector'] != needed:
        raise ValueError('Synchronized Hips frames must equal the fullbody/feet frame union')
    return dict(backend=PELVIS_BACKEND, constraintCounts={kind:len(frames) for kind,frames in frames_by_type.items()},
                maximumSharedFramePositionErrorMeters=maximum_shared_position_error,
                maximumSharedFrameRotationErrorRadians=maximum_shared_rotation_error)


def _rotation_between(source, target):
    source = source / np.linalg.norm(source)
    target = target / np.linalg.norm(target)
    cosine = float(np.clip(source @ target, -1., 1.))
    cross = np.cross(source, target)
    sine = float(np.linalg.norm(cross))
    if sine < 1e-12:
        if cosine > 0:
            return np.eye(3)
        basis = np.eye(3)[int(np.argmin(np.abs(source)))]
        axis = np.cross(source, basis)
        return Rotation.from_rotvec(axis / np.linalg.norm(axis) * np.pi).as_matrix()
    return Rotation.from_rotvec(cross / sine * math.atan2(sine, cosine)).as_matrix()


def solve_foot_ik(local, root, rig, side, ankle_target, foot_rotation, reference_toes, pole_direction=None):
    """Two-link positional IK, preserving the knee pole and world foot twist.

    Input/output use SOMA local rotation matrices. Unreachable anchors raise;
    no limb stretching, root teleport or silent target clipping is permitted.
    """
    names = {name: index for index, name in enumerate(rig['jointNames'])}
    prefix = side.title()
    thigh, shin, foot = (names[prefix + suffix] for suffix in ('Leg', 'Shin', 'Foot'))
    if rig['parents'][shin] != thigh or rig['parents'][foot] != shin:
        raise ValueError('Expected a two-link SOMA leg chain')
    result = np.asarray(local).copy()
    global_rot, positions = fk(result[None], np.asarray(root)[None], rig)
    rotations, positions = global_rot[0], positions[0]
    hip, knee, ankle = positions[[thigh, shin, foot]]
    upper, lower = float(np.linalg.norm(knee - hip)), float(np.linalg.norm(ankle - knee))
    target = np.asarray(ankle_target, dtype=float)
    direction = target - hip
    distance = float(np.linalg.norm(direction))
    if distance > upper + lower - 1e-7 or distance < abs(upper - lower) + 1e-7:
        raise ValueError(f'{side} ankle anchor is unreachable: {distance:.6f}m, '
                         f'leg interval ({abs(upper-lower):.6f}, {upper+lower:.6f})m')
    direction /= distance
    pole = knee - hip - direction * float((knee - hip) @ direction)
    if pole_direction is not None:
        requested=np.asarray(pole_direction,dtype=float)
        if requested.shape!=(3,) or not np.isfinite(requested).all() or np.linalg.norm(requested)<1e-8:
            raise ValueError('Invalid knee pole direction')
        pole=requested-direction*float(requested@direction)
    if np.linalg.norm(pole) < 1e-8:
        # A straight source knee has no pole. Use the source body's forward
        # direction, or a deterministic perpendicular only if it is parallel.
        pole = rotations[0, :, 2] - direction * float(rotations[0, :, 2] @ direction)
    if np.linalg.norm(pole) < 1e-8:
        basis = np.eye(3)[int(np.argmin(np.abs(direction)))]
        pole = basis - direction * float(basis @ direction)
    pole /= np.linalg.norm(pole)
    along = (upper * upper - lower * lower + distance * distance) / (2 * distance)
    bend = math.sqrt(max(0., upper * upper - along * along))
    solved_knee = hip + direction * along + pole * bend
    thigh_global = _rotation_between(knee - hip, solved_knee - hip) @ rotations[thigh]
    shin_global = _rotation_between(ankle - knee, target - solved_knee) @ rotations[shin]
    parent = rig['parents'][thigh]
    result[thigh] = rotations[parent].T @ thigh_global if parent >= 0 else thigh_global
    result[shin] = thigh_global.T @ shin_global
    result[foot] = shin_global.T @ foot_rotation
    # Kimodo expands Left/RightFoot to toe positions as well. Freeze the toe
    # chain's local pose within the same support pair, not just its ankle.
    for index, rotation in reference_toes.items():
        result[index] = rotation
    actual_rot, actual_positions = fk(result[None], np.asarray(root)[None], rig)
    if np.linalg.norm(actual_positions[0, foot] - target) > 1e-7:
        raise ValueError('Leg IK failed ankle-position reconstruction')
    if not np.allclose(actual_rot[0, foot], foot_rotation, atol=1e-8):
        raise ValueError('Leg IK failed foot-orientation reconstruction')
    return result


def _support_runs(mask):
    edges = np.diff(np.r_[False, mask, False].astype(int))
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))


def materialize_gait_constraints(params, reference_motion, rig, *, phase_evidence=None):
    """Return actual Kimodo JSON constraints and an auditable input receipt.

    Required params: seconds, fps, targetPeriodSeconds, rootPath with kind=linear,
    positive velocity, unit direction=[x,z], optional headingDegrees (constant).
    Optional: gaitPolicy, supportPairSpacingSeconds=.1,
    maxConstraintFramesPerType=19, maximumHipLoweringMeters=.03.
    A complete observed left/right cycle is
    mandatory, but acceptance of that source as a master is never implied.
    """
    validate_motion(reference_motion, rig)
    seconds, fps = float(params['seconds']), float(params['fps'])
    period = float(params['targetPeriodSeconds'])
    spacing = float(params.get('supportPairSpacingSeconds', .1))
    support_fraction = float(params.get('supportWindowFraction', .5))
    maximum_frames = params.get('maxConstraintFramesPerType', 19)
    maximum_drop = float(params.get('maximumHipLoweringMeters', .03))
    maximum_anchor_correction=float(params.get('maximumAnchorCorrectionMeters',math.inf))
    if not all(math.isfinite(value) and value > 0 for value in (seconds, fps, period, spacing)):
        raise ValueError('Gait duration, FPS, period and support spacing must be positive')
    if not isinstance(maximum_frames, int) or not 1 <= maximum_frames < 20:
        raise ValueError('Kimodo gait conditions must use fewer than 20 frames per sparse type')
    if not math.isfinite(maximum_drop) or maximum_drop < 0:
        raise ValueError('Maximum input-pose hip lowering must be finite and nonnegative')
    if not math.isfinite(support_fraction) or not 0<support_fraction<=1:
        raise ValueError('Support window fraction must be between zero and one')
    if maximum_anchor_correction<=0 or math.isnan(maximum_anchor_correction):
        raise ValueError('Maximum support anchor correction must be positive')
    count = int(seconds * fps)
    period_frames = period * fps
    if count < 3 or period_frames < 4 or not math.isclose(period_frames, round(period_frames), abs_tol=1e-6):
        raise ValueError('Target gait period must span an integral number of at least four frames')
    period_frames = round(period_frames)
    path = params['rootPath']
    if path['kind'] != 'linear':
        raise ValueError('Steady gait refinement currently requires a linear root path')
    velocity = float(path['velocity'])
    direction = np.asarray(path['direction'], dtype=float)
    if direction.shape != (2,) or not np.isfinite(direction).all() or not np.isclose(np.linalg.norm(direction), 1., atol=1e-6):
        raise ValueError('The root path direction must be a finite unit XZ vector')
    if not math.isfinite(velocity) or velocity <= 0:
        raise ValueError('The root path velocity must be positive')
    heading = math.radians(float(path.get('headingDegrees', 0.)))
    if not math.isfinite(heading):
        raise ValueError('Invalid root path heading')
    preserve_heading = params.get('preserveReferenceHeading', False)
    if not isinstance(preserve_heading, bool):
        raise ValueError('preserveReferenceHeading must be a boolean')
    preserve_pelvis = params.get('preserveReferencePelvisPath', False)
    if not isinstance(preserve_pelvis, bool):
        raise ValueError('preserveReferencePelvisPath must be a boolean')
    if preserve_pelvis and (phase_evidence is None or not preserve_heading):
        raise ValueError('Preserving native pelvis path requires verified native phase evidence and heading')
    if preserve_pelvis and params.get('constrainHandOrientations', False):
        raise ValueError('Preserved pelvis locomotion constraints cannot also pin the hands')
    if preserve_pelvis and params.get('pelvisConstraintBackend', PELVIS_BACKEND) != PELVIS_BACKEND:
        raise ValueError('Unsupported preserved pelvis constraint backend')
    pose_mode = params.get('poseConstraintMode', 'full_body')
    if pose_mode not in ('full_body', 'feet_only', 'lower_body'):
        raise ValueError('Unknown gait pose constraint mode')
    if pose_mode != 'full_body' and params.get('constrainHandOrientations', False):
        raise ValueError('Lower-body gait constraints cannot also pin the hands')
    audit = audit_candidate(reference_motion, rig, policy=params.get('gaitPolicy'))
    receipt = {'schemaVersion': 1, 'scope': 'generation_constraint_consistency_only',
               'referenceAcceptedAsMaster': False, 'referenceAuditPass': audit['pass'],
               'referenceMotionSha256': audit['sourceMotionSha256'],
               'rigSha256': audit['rigSha256'], 'referenceAudit': audit,
               'parameters': params, 'pass': False}
    phase_detection = audit['detection']
    phase_cycles = audit['cycles']
    if phase_evidence is not None:
        from .constraint_phase import validate_native_phase_evidence
        verified = validate_native_phase_evidence(phase_evidence, reference_motion, rig)
        if (not math.isclose(verified['expectedSpeedMps'], velocity, abs_tol=1e-8, rel_tol=0)
                or not math.isclose(verified['targetPeriodSeconds'], period, abs_tol=1e-8, rel_tol=0)
                or not np.allclose(verified['travelDirectionXZ'], direction, atol=1e-8, rtol=0)):
            raise GaitConstraintError('Native phase evidence differs from requested speed, period or direction', receipt)
        phase_detection, phase_cycles = verified['detection'], verified['cycles']
        receipt['nativePhaseEvidence'] = dict(policy=verified['policy'], sha256=verified['sha256'],
            source=verified['source'], sourceAuditPass=verified['sourceAuditPass'],
            scope=verified['scope'], referenceAcceptedAsMaster=False)
    native_pelvis_residual = None
    if preserve_pelvis:
        # Kimodo's smooth_root_2d and actual Hips translation are distinct.
        # Verify the residual against immutable ALS Pelvis geometry, rather
        # than permitting an arbitrary offset or changing the path tolerance.
        from .als_reference import original_input
        from .graph import read_json
        from .publish import world_pose
        native_index, native_row, native_payload_path = original_input(verified['indexPath'], verified['source'])
        native_raw = read_json(native_payload_path)['raw']
        native_skeleton = native_index['skeletons']['/ALS/ALS/Character/SK_Als.SK_Als']
        native_world = world_pose(native_raw, native_skeleton)
        native_period = native_raw['sampledKeyCount'] - 1
        if native_period != period_frames or not math.isclose(float(reference_motion['fps']), fps, abs_tol=1e-8, rel_tol=0):
            raise GaitConstraintError('Native pelvis reference differs from requested sampling domain', receipt)
        frame_numbers = np.arange(len(reference_motion['root_positions']))
        native_pelvis_residual = native_world['pelvis'][1][frame_numbers % native_period, :2] * .01
        reference_path = frame_numbers[:, None] / fps * velocity * direction
        observed_residual = reference_motion['root_positions'][:, [0, 2]] - reference_path
        native_residual_error = float(np.max(np.abs(observed_residual-native_pelvis_residual)))
        if native_residual_error > 1e-8:
            receipt['nativePelvisResidualErrorMeters'] = native_residual_error
            raise GaitConstraintError('Reference pelvis offset differs from immutable native geometry', receipt)
        receipt['nativePelvisPath'] = dict(source=verified['source'], sourcePayloadSha256=native_row['sha256'],
            referenceMotionSha256=verified['referenceMotionSha256'],
            smoothRootOriginXZ=[0., 0.], residualCoordinates='world XZ meters',
            maximumResidualMeters=float(np.linalg.norm(native_pelvis_residual, axis=1).max()),
            nativeResidualErrorMeters=native_residual_error)
    candidates = [cycle for cycle in phase_cycles if len(cycle['oppositeTouchdowns']) == 1
                  and cycle['oppositePhase'] is not None and .2 < cycle['oppositePhase'] < .8]
    if not candidates:
        raise GaitConstraintError('Reference contains no complete alternating same-foot cycle', receipt)
    # Prefer the observed cycle closest to the original half-cycle contract,
    # then actual unmodified seam quality. Its remaining failures are retained.
    cycle = min(candidates, key=lambda item: (abs(math.log(item['durationSeconds']/period)),
                                              sum(not check['pass'] for check in item['checks']),
                                              abs(item['oppositePhase'] - .5),
                                              item.get('seamBodyMeters', math.inf), item['start']))
    start, end = cycle['start'], cycle['endInclusive']
    source_frames = np.arange(start, end + 1)
    local_source = reference_motion['local_rot_mats'][start:end + 1]
    root_source = reference_motion['root_positions'][start:end + 1]
    def pelvis_residual_at(frame):
        source = start + (frame % period_frames) / period_frames * (end-start)
        return np.array([np.interp(source, source_frames, native_pelvis_residual[start:end+1, axis])
                         for axis in range(2)])
    interpolators = [Slerp(source_frames, Rotation.from_matrix(local_source[:, bone]))
                     for bone in range(len(rig['jointNames']))]
    names = {name: index for index, name in enumerate(rig['jointNames'])}
    hips = rig.get('hipJointNames', ['RightLeg', 'LeftLeg'])
    right_hip, left_hip = (names[name] for name in hips)
    world_path = np.arange(count)[:, None] / fps * velocity * direction
    desired_headings = np.full(count, heading)
    if preserve_heading:
        # Pelvis yaw is part of the planted-leg geometry, independently of the
        # travel direction. Give root2d the same phase-varying yaw as the poses.
        sample_times = start + (np.arange(count) % period_frames) / period_frames * (end - start)
        sampled_local = np.stack([interpolator(sample_times).as_matrix()
                                  for interpolator in interpolators], axis=1)
        _, sampled_points = fk(sampled_local, np.zeros((count, 3)), rig)
        line = sampled_points[:, right_hip] - sampled_points[:, left_hip]
        desired_headings = np.arctan2(line[:, 2], -line[:, 0])

    def required_lowering(local, root, side, anchor):
        _, joints = fk(local[None], root[None], rig)
        thigh, shin, foot = (names[side.title() + suffix] for suffix in ('Leg', 'Shin', 'Foot'))
        hip = joints[0, thigh]
        reach = (np.linalg.norm(joints[0, shin] - hip) +
                 np.linalg.norm(joints[0, foot] - joints[0, shin]) - 2e-6)
        horizontal_squared = float(np.sum((hip - anchor)[[0, 2]]**2))
        if horizontal_squared >= reach * reach:
            return math.inf
        vertical_limit = math.sqrt(reach * reach - horizontal_squared)
        vertical = float(hip[1] - anchor[1])
        if vertical < -vertical_limit:
            return math.inf
        return max(0., vertical - vertical_limit)

    stride_scale = velocity * period / np.linalg.norm(root_source[-1, [0, 2]] - root_source[0, [0, 2]])
    if not math.isfinite(stride_scale) or stride_scale <= 0:
        raise GaitConstraintError('Reference has no measurable cycle travel', receipt)
    stride_means = {}
    if params.get('matchStrideToPeriod'):
        normalized = local_source.copy()
        reference_globals, reference_points = fk(normalized, root_source, rig)
        hip_line = reference_points[:, right_hip] - reference_points[:, left_hip]
        angles = np.arctan2(hip_line[:, 2], -hip_line[:, 0])
        if not preserve_heading:
            normalized[:, 0] = Rotation.from_euler('y', (heading-angles)[:, None]).as_matrix() @ normalized[:, 0]
        _, reference_points = fk(normalized, root_source, rig)
        for side in ('left', 'right'):
            foot = names[side.title()+'Foot']
            relative = reference_points[:-1, foot][:, [0, 2]] - root_source[:-1, [0, 2]]
            stride_means[side] = float(np.mean(relative @ direction))

    def sample_pose(frame):
        phase = (frame % period_frames) / period_frames
        source = start + phase * (end - start)
        local = np.stack([interpolator([source]).as_matrix()[0] for interpolator in interpolators])
        root = np.array([frame / fps * velocity * direction[0],
                         np.interp(source, source_frames, root_source[:, 1]),
                         frame / fps * velocity * direction[1]])
        if preserve_pelvis:
            root[[0, 2]] += pelvis_residual_at(frame)
        _, points = fk(local[None], root[None], rig)
        hip_line = points[0, right_hip] - points[0, left_hip]
        measured_heading = math.atan2(hip_line[2], -hip_line[0])
        if not preserve_heading:
            local[0] = Rotation.from_euler('y', heading - measured_heading).as_matrix() @ local[0]
        if stride_means:
            global_rot, points = fk(local[None], root[None], rig)
            targets = {}
            for side in ('left', 'right'):
                foot = names[side.title()+'Foot']
                relative = points[0, foot, [0, 2]] - root[[0, 2]]
                correction = (stride_scale-1.) * (float(relative @ direction)-stride_means[side])
                if abs(correction) > .15:
                    raise GaitConstraintError('Reference stride scaling exceeds the 15 cm correction bound', receipt)
                target = points[0, foot].copy()
                target[[0, 2]] += correction * direction
                targets[side] = target
            drop = max(required_lowering(local, root, side, target) for side, target in targets.items())
            if drop > maximum_drop:
                raise GaitConstraintError('Reference stride scaling exceeds the hip-lowering bound', receipt)
            root[1] -= drop
            for side, target in targets.items():
                foot = names[side.title()+'Foot']
                toes = {}
                for bone, parent in enumerate(rig['parents']):
                    ancestor = parent
                    while ancestor >= 0 and ancestor != foot:
                        ancestor = rig['parents'][ancestor]
                    if ancestor == foot: toes[bone] = local[bone].copy()
                try:
                    local = solve_foot_ik(local, root, rig, side, target, global_rot[0, foot], toes)
                except ValueError as error:
                    raise GaitConstraintError('Reference stride scaling is unreachable: '+str(error), receipt) from error
        return local, root

    boundaries = list(range(0, count, period_frames))
    maximum_cycles = params.get('maximumConstraintCycles')
    if maximum_cycles is not None:
        if not isinstance(maximum_cycles, int) or maximum_cycles < 2 or preserve_pelvis and isinstance(maximum_cycles, bool):
            raise ValueError('At least two complete constrained cycles are required')
        boundaries = boundaries[:maximum_cycles + 1]
    requested_cycles = None
    if preserve_pelvis:
        available_cycles = (count-1)//period_frames
        requested_cycles = maximum_cycles if maximum_cycles is not None else available_cycles
        if requested_cycles < 2 or requested_cycles > available_cycles:
            raise GaitConstraintError('Requested native pelvis constraint cycles do not fit the clip', receipt)
        boundaries = [cycle*period_frames for cycle in range(requested_cycles+1)]
    cycle_boundaries = boundaries.copy()
    # The final closing boundary is not the start of another constrained cycle.
    # Keep historical behavior for the default mode to avoid cache/recipe churn.
    cycle_starts = cycle_boundaries[:-1] if preserve_pelvis else cycle_boundaries
    boundary_context = params.get('boundaryContextFrames', 0)
    if not isinstance(boundary_context, int) or not 0 <= boundary_context < period_frames // 2:
        raise ValueError('Boundary context must be a nonnegative frame radius below half a period')
    if boundary_context:
        # Constrain matching pose neighborhoods across interior cycle seams.
        # These supply motion context, not a native velocity-constraint API.
        anchors = set(boundaries)
        for boundary in boundaries[1:-1]:
            anchors.update(range(boundary-boundary_context, boundary+boundary_context+1))
        boundaries = sorted(anchors)
    phases = params.get('fullBodyPhaseFractions', [])
    if any(not math.isfinite(phase) or not 0 < phase < 1 for phase in phases):
        raise ValueError('Interior gait pose phases must be between zero and one')
    phase_cycles = params.get('fullBodyPhaseCycles', 1)
    if not isinstance(phase_cycles, int) or phase_cycles < 1:
        raise ValueError('Interior pose cycle count must be a positive integer')
    boundaries = sorted(set(boundaries) | {boundary+round(phase*period_frames)
        for boundary in cycle_starts[:phase_cycles] for phase in phases
        if boundary+round(phase*period_frames) < count})
    # Constrain the actual last frame at its own phase as well. The inclusive
    # clip endpoint is rarely an integral number of periods; do not give it
    # the phase-zero pose and create a shortened final stride.
    if params.get('constrainClipEndpoint', False):
        boundaries = sorted(set(boundaries) | {count - 1})
    foot_boundary_radius = params.get('footBoundaryContextFrames')
    foot_boundaries = set()
    if foot_boundary_radius is not None:
        if (not isinstance(foot_boundary_radius, int) or isinstance(foot_boundary_radius, bool)
                or not 0 <= foot_boundary_radius < period_frames // 2):
            raise ValueError('Foot boundary context must be a nonnegative frame radius below half a period')
        for boundary in cycle_boundaries:
            foot_boundaries.update(range(max(0, boundary-foot_boundary_radius),
                                         min(count, boundary+foot_boundary_radius+1)))
        if params.get('constrainClipEndpoint', False):
            foot_boundaries.add(count-1)
    if pose_mode != 'full_body':
        # Reuse the measured phase targets on the feet while leaving the torso,
        # head and arms to the text-conditioned model. Kimodo's foot constraint
        # decodes only the corresponding end effector from these pose payloads.
        foot_boundaries.update(boundaries)
    poses = {frame: sample_pose(frame) for frame in boundaries}
    for frame in foot_boundaries:
        if frame not in poses:poses[frame] = sample_pose(frame)
    side_frames = {side: foot_boundaries.copy() for side in ('left', 'right')}
    pairs, targets = [], {}
    anchor_count = params.get('supportAnchorCount', 2)
    if not isinstance(anchor_count, int) or not 2 <= anchor_count <= maximum_frames:
        raise ValueError('A support interval requires at least two sparse anchors')

    contacts = np.asarray(phase_detection['contacts'])[start:end]
    for index, side in enumerate(('left', 'right')):
        runs = _support_runs(contacts[:, index])
        if not runs:
            raise GaitConstraintError('Reference has no support for ' + side, receipt)
        support_start, support_end = max(runs, key=lambda run: run[1] - run[0])
        # Use two interior samples of the measured contact interval. Touchdown
        # and liftoff remain free for the model to generate natural transitions.
        first_phase = support_start / (end - start)
        last_phase = (support_end - 1) / (end - start)
        midpoint = (first_phase + last_phase) * period_frames / 2
        half_width = min(spacing * fps / 2, (last_phase - first_phase) * period_frames * support_fraction / 2)
        if half_width < .5:
            raise GaitConstraintError('Reference support window too short for two sparse anchors: ' + side, receipt)
        pair_frames = max(1, int(math.floor(2 * half_width + 1e-9)))
        foot = names[side.title() + 'Foot']
        # Search the measured support interval without shortening the requested
        # pair spacing. Prefer the least hip adjustment, then its midpoint.
        lower = math.ceil(first_phase * period_frames)
        upper = math.floor(last_phase * period_frames)
        if upper - lower >= pair_frames + 2:
            lower, upper = lower + 1, upper - 1
        alternatives = []
        for candidate_first in range(lower, upper - pair_frames + 1):
            candidate_second = candidate_first + pair_frames
            center = (candidate_first + candidate_second) / 2
            center_local, center_root = sample_pose(center)
            _, center_positions = fk(center_local[None], center_root[None], rig)
            anchor = center_positions[0, foot]
            anchor_frames = sorted(set(np.rint(np.linspace(candidate_first,
                candidate_second, anchor_count)).astype(int).tolist()))
            drops,corrections = [],[]
            for frame in anchor_frames:
                candidate_local,candidate_root=sample_pose(frame)
                _,candidate_points=fk(candidate_local[None],candidate_root[None],rig)
                corrections.append(float(np.linalg.norm(candidate_points[0,foot]-anchor)))
                drops.append(required_lowering(candidate_local,candidate_root,side,anchor))
            alternatives.append({'first': candidate_first, 'second': candidate_second,
                                 'frames': anchor_frames,
                                 'maximumAnchorCorrectionMeters':max(corrections),
                                 'maximumRequiredDropMeters': max(drops) if np.isfinite(drops).all() else None,
                                 'distanceFromSupportMidpointFrames': abs(center - midpoint)})
        feasible = [item for item in alternatives if item['maximumRequiredDropMeters'] is not None
                    and item['maximumRequiredDropMeters'] <= maximum_drop
                    and item['maximumAnchorCorrectionMeters']<=maximum_anchor_correction]
        if not feasible:
            receipt.update(selectedReferenceCycle=cycle, supportPairs=pairs,
                           failedSide=side, supportPairSearch=alternatives,
                           maximumHipLoweringMeters=maximum_drop)
            raise GaitConstraintError('No reachable support pair within the hip-lowering limit: ' + side, receipt)
        choice = min(feasible, key=lambda item: (item['maximumRequiredDropMeters'],
                                                item['distanceFromSupportMidpointFrames'], item['first']))
        base_first, base_second = choice['first'], choice['second']
        for boundary in cycle_starts:
            first, second = boundary + base_first, boundary + base_second
            if second >= count:
                continue
            anchor_frame = (first + second) / 2
            reference_local, reference_root = sample_pose(anchor_frame)
            reference_global, reference_positions = fk(reference_local[None], reference_root[None], rig)
            anchor = reference_positions[0, foot]
            orientation = reference_global[0, foot]
            toe_indices = {}
            for bone in range(len(rig['parents'])):
                ancestor = rig['parents'][bone]
                while ancestor >= 0 and ancestor != foot:
                    ancestor = rig['parents'][ancestor]
                if ancestor == foot:
                    toe_indices[bone] = reference_local[bone].copy()
            anchor_frames = [boundary+frame for frame in choice['frames']]
            pair = {'side': side, 'cycleBoundaryFrame': boundary, 'frames': anchor_frames,
                    'anchorMeters': anchor.tolist(), 'footGlobalRotation': orientation.tolist(),
                    'rootPositions': [], 'maximumPositionErrorMeters': 0.,
                    'maximumRotationErrorRadians': 0., 'supportPairSearch': alternatives}
            for frame in anchor_frames:
                side_frames[side].add(frame)
                targets.setdefault(frame, []).append({'side': side, 'anchor': anchor,
                    'orientation': orientation, 'toes': toe_indices, 'pair': pair})
            pairs.append(pair)
    counts = {side + '-foot': len(frames) for side, frames in side_frames.items()}
    if pose_mode == 'full_body':
        counts['fullbody'] = len(boundaries)
    elif pose_mode == 'lower_body':
        counts['end-effector'] = len(boundaries)
    pelvis_frames = sorted(set(boundaries) | set().union(*side_frames.values())) if preserve_pelvis else []
    if preserve_pelvis:
        counts['end-effector'] = len(pelvis_frames)
    constrain_hands=bool(params.get('constrainHandOrientations',False))
    if constrain_hands:
        if any(side+'Hand' not in names for side in ('Left','Right')):
            raise ValueError('Hand orientation constraints require both hand joints')
        counts.update({'left-hand':len(boundaries),'right-hand':len(boundaries)})
    if any(value > maximum_frames for value in counts.values()):
        receipt.update(constraintCounts=counts, selectedReferenceCycle=cycle)
        raise GaitConstraintError('Sparse constraint frame budget exceeded', receipt)
    if any(value == 0 for value in counts.values()):
        raise GaitConstraintError('A gait condition type has no valid frame', receipt)
    adjustments = []
    for frame, frame_targets in sorted(targets.items()):
        local, root = sample_pose(frame)
        original_height = float(root[1])
        lowering = max(required_lowering(local, root, target['side'], target['anchor'])
                       for target in frame_targets)
        if lowering > maximum_drop:
            receipt.update(failedFrame=frame, requiredHipLoweringMeters=lowering if math.isfinite(lowering) else None,
                           maximumHipLoweringMeters=maximum_drop, supportPairs=pairs)
            raise GaitConstraintError('Shared hip height cannot satisfy all foot anchors', receipt)
        source_frame = start+(frame % period_frames)/period_frames*(end-start)
        reference_height = np.interp(source_frame, source_frames, root_source[:, 1])
        if reference_height-root[1]+lowering > maximum_drop+1e-8:
            raise GaitConstraintError('Combined stride and support hip lowering exceeds its bound', receipt)
        root[1] -= lowering
        try:
            for target in frame_targets:
                local = solve_foot_ik(local, root, rig, target['side'], target['anchor'],
                                      target['orientation'], target['toes'])
        except ValueError as exc:
            receipt.update(selectedReferenceCycle=cycle, supportPairs=pairs, failure=str(exc), failedFrame=frame)
            raise GaitConstraintError(str(exc), receipt) from exc
        poses[frame] = local, root
        adjustments.append({'frame': frame, 'originalHipHeightMeters': original_height,
                            'constrainedHipHeightMeters': float(root[1]), 'dropMeters': lowering})
        actual_global, actual_positions = fk(local[None], root[None], rig)
        for target in frame_targets:
            foot = names[target['side'].title() + 'Foot']
            pair = target['pair']
            pair['rootPositions'].append(root.tolist())
            pair['maximumPositionErrorMeters'] = max(pair['maximumPositionErrorMeters'],
                float(np.linalg.norm(actual_positions[0, foot] - target['anchor'])))
            angle = np.linalg.norm(Rotation.from_matrix(actual_global[0, foot] @ target['orientation'].T).as_rotvec())
            pair['maximumRotationErrorRadians'] = max(pair['maximumRotationErrorRadians'], float(angle))
    constraints = [{'type': 'root2d', 'frame_indices': list(range(count)),
                    'smooth_root_2d': world_path.tolist(),
                    'global_root_heading': np.column_stack((np.cos(desired_headings),
                                                             np.sin(desired_headings))).tolist()}]
    maximum_path_error, maximum_heading_error, maximum_pelvis_error = 0., 0., 0.
    sparse = ([('fullbody', boundaries)] if pose_mode == 'full_body' else [])
    if pose_mode == 'lower_body' and not preserve_pelvis:
        sparse.append(('end-effector', boundaries))
    sparse += [(side+'-foot',sorted(side_frames[side])) for side in ('left','right')]
    if preserve_pelvis:
        # Backend postprocess recognizes named foot masks, but generic multi-EE
        # names do not set those masks. Keep the named feet and append actual
        # Hips on their union LAST. It then restores actual Hips after foot EE
        # extraction temporarily substitutes smooth_root_2d; fullbody is sorted
        # last by the backend and reads the identical pose at shared frames.
        sparse.append(('end-effector', pelvis_frames))
    if constrain_hands:sparse.extend((side+'-hand',boundaries) for side in ('left','right'))
    for kind,frames in sparse:
        local = np.stack([poses[frame][0] for frame in frames])
        root = np.stack([poses[frame][1] for frame in frames])
        axis = Rotation.from_matrix(local.reshape(-1, 3, 3)).as_rotvec().reshape(len(frames), len(names), 3)
        validate_motion({'local_rot_mats': local, 'root_positions': root, 'fps': fps}, rig)
        _, positions = fk(local, root, rig)
        line = positions[:, right_hip] - positions[:, left_hip]
        observed_heading = np.arctan2(line[:, 2], -line[:, 0])
        difference = observed_heading - desired_headings[frames]
        angle_error = np.abs(np.arctan2(np.sin(difference), np.cos(difference)))
        maximum_heading_error = max(maximum_heading_error, float(angle_error.max()))
        expected_residual = np.stack([pelvis_residual_at(frame) for frame in frames]) if preserve_pelvis else 0.
        reconstructed_smooth_root = root[:, [0, 2]] - expected_residual
        maximum_path_error = max(maximum_path_error, float(np.max(np.abs(reconstructed_smooth_root - world_path[frames]))))
        if preserve_pelvis:
            maximum_pelvis_error = max(maximum_pelvis_error,
                float(np.max(np.abs((root[:, [0, 2]]-world_path[frames])-expected_residual))))
        constraints.append({'type': kind, 'frame_indices': frames, 'local_joints_rot': axis.tolist(),
                            'root_positions': root.tolist(), 'smooth_root_2d': world_path[frames].tolist()})
        if kind == 'end-effector':
            constraints[-1]['joint_names'] = ['Hips']
    if maximum_path_error > 1e-8 or maximum_heading_error > 1e-7:
        receipt.update(maximumPathErrorMeters=maximum_path_error, maximumHeadingErrorRadians=maximum_heading_error)
        raise GaitConstraintError('Foot/fullbody constraints contradict root2d path or heading', receipt)
    layout = None
    if preserve_pelvis:
        try:
            layout = validate_pelvis_constraint_consistency(constraints, rig, world_path,
                np.stack([pelvis_residual_at(frame) for frame in range(count)]), desired_headings,
                maximum_frames=maximum_frames)
        except ValueError as error:
            receipt['failure'] = str(error)
            raise GaitConstraintError('Preserved pelvis constraint consistency failed: '+str(error), receipt) from error
    receipt['pass'] = True
    receipt.update(selectedReferenceCycle=cycle, targetPeriodFrames=period_frames,
                   poseConstraintMode=pose_mode,
                   upperBodyConstrained=pose_mode == 'full_body' or constrain_hands,
                   referenceStrideScale=stride_scale,
                   strideMatchedToPeriod=bool(stride_means),
                   preserveReferenceHeading=preserve_heading,
                   rootHeadingRangeDegrees=np.rad2deg([desired_headings.min(), desired_headings.max()]).tolist(),
                   boundaryFrames=boundaries, cycleBoundaryFrames=cycle_boundaries,
                   footBoundaryFrames=sorted(foot_boundaries),
                   supportPairs=pairs, constraintCounts=counts,
                   maximumPathErrorMeters=maximum_path_error,
                   maximumHeadingErrorRadians=maximum_heading_error,
                   hipHeightAdjustments=adjustments, maximumHipLoweringMeters=maximum_drop,
                   constraintsSha256=hashlib.sha256(json.dumps(constraints, sort_keys=True,
                       separators=(',', ':')).encode()).hexdigest(),
                   requires=['generated_candidate_authoring_gate', 'target_skin_gate', 'runtime_gate'])
    if preserve_pelvis:
        receipt.update(preserveReferencePelvisPath=True, pelvisConstrained=True,
            maximumPelvisResidualErrorMeters=maximum_pelvis_error,
            pathErrorMeasurement='actual Hips XZ minus verified native residual versus specified smooth_root_2d',
            pelvisConstraintFrames=pelvis_frames, requestedConstraintCycles=requested_cycles,
            completeConstrainedCycles=len(cycle_starts), cycleStartFrames=cycle_starts,
            pelvisConstraintPackaging='named feet followed by synchronized Hips EE; backend named foot masks retained',
            pelvisConstraintConsistency=layout)
    return constraints, receipt

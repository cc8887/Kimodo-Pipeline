"""Chest-relative fixed upper arms, swinging forearms and stance-aware backs."""
import math

import numpy as np
from scipy.spatial.transform import Rotation

from .gait import _array_hash
from .motion import complete, fk, validate_motion

POLICY = 'locomotion-idle-forearm-v2'
SPINE = ('Spine1', 'Spine2', 'Chest')
NECK = ('Neck1', 'Neck2', 'Head')
ARMS = tuple(s + j for s in ('Left', 'Right') for j in ('Shoulder', 'Arm', 'ForeArm', 'Hand'))
LIMITS = {'torsoForwardDegrees', 'neckForwardDegrees', 'headDownDegrees',
          'armAbductionDegrees', 'upperArmMotionDegrees', 'minimumHandTravelMeters'}


def validate_profile(profile):
    keys = {'policy', 'bodyMotionGain', 'forearmMotionGain', 'forearmSwingDegrees',
            'elbowRestDegrees', 'referenceFraction', 'stanceMode', 'forearmSwing', 'limits'}
    if set(profile) != keys or profile['policy'] != POLICY:
        raise ValueError('Unknown forearm locomotion profile')
    for key in ('bodyMotionGain', 'forearmMotionGain', 'referenceFraction'):
        value = profile[key]
        if isinstance(value, bool) or not math.isfinite(value) or not 0 < value < 1:
            raise ValueError('Forearm posture gains must be between zero and one')
    for key in ('forearmSwingDegrees', 'elbowRestDegrees'):
        value = profile[key]
        if isinstance(value, bool) or not math.isfinite(value) or not 0 < value < 90:
            raise ValueError('Forearm angles must be between zero and ninety degrees')
    if profile['stanceMode'] not in ('standing', 'crouching', 'transition', 'mixed') or type(profile['forearmSwing']) is not bool:
        raise ValueError('Invalid stance or forearm swing mode')
    if set(profile['limits']) != LIMITS or any(isinstance(v, bool) or not math.isfinite(v) or v <= 0 for v in profile['limits'].values()):
        raise ValueError('Invalid forearm posture limits')


def context_profile(profile, semantic, family=None):
    result = dict(profile, limits=dict(profile['limits']))
    result['stanceMode'] = ('crouching' if semantic.startswith('Crouch') and semantic != 'CrouchToStand'
                           else 'transition' if semantic in ('StandToCrouch', 'CrouchToStand')
                           else 'mixed' if semantic.endswith('_Poses') else 'standing')
    result['forearmSwing'] = family == 'gait_cycle'
    result['elbowRestDegrees'] = 60. if semantic.startswith(('Run', 'Sprint', 'Crouch')) else 45.
    return result


def author_idle_arms(source, rig, profile):
    """Remove the breathing overlay's humeral swing without altering its body."""
    validate_profile(profile); validate_motion(source, rig)
    names={n:i for i,n in enumerate(rig['jointNames'])}
    original,_=fk(source['local_rot_mats'],source['root_positions'],rig)
    frame=round(profile['referenceFraction']*(len(original)-1));chest=names['Chest']
    local=source['local_rot_mats'].copy();global_out=np.empty_like(original)
    edited={side+joint for side in ('Left','Right') for joint in ('Shoulder','Arm')}
    for i,parent in enumerate(rig['parents']):
        if rig['jointNames'][i] in edited:
            desired=global_out[:,chest]@(original[frame,chest].T@original[frame,i])
            local[:,i]=global_out[:,parent].swapaxes(-1,-2)@desired
        global_out[:,i]=local[:,i] if parent<0 else global_out[:,parent]@local[:,i]
    untouched=[i for i,n in enumerate(rig['jointNames']) if n not in edited]
    if not np.array_equal(local[:,untouched],source['local_rot_mats'][:,untouched]):
        raise ValueError('Idle arm edit changed breathing, forearms or lower body')
    result=complete(local,source['root_positions'].copy(),rig,float(source['fps']))
    return result,dict(policy=POLICY,profile=profile,referenceFrame=frame,
        inputMotionSha256=_array_hash(source['local_rot_mats'],source['root_positions'],source['fps']),
        outputMotionSha256=_array_hash(result['local_rot_mats'],result['root_positions'],result['fps']),
        preservedRootAndLowerBody=True,preservedIdleBodyAndForearms=True,modifiedJoints=sorted(edited),
        scope='Idle breathing body and forearms retained; upper-arm swing removed from the additive source')


def author_posture(source, idle, rig, profile):
    from .locomotion_style import _heading, measure_posture
    validate_profile(profile)
    validate_motion(source, rig); validate_motion(idle, rig)
    names = {n: i for i, n in enumerate(rig['jointNames'])}
    if set(SPINE + NECK + ARMS) - set(names):
        raise ValueError('Forearm authoring requires the complete SOMA upper-body chains')
    original, points = fk(source['local_rot_mats'], source['root_positions'], rig)
    reference, _ = fk(idle['local_rot_mats'], idle['root_positions'], rig)
    frame = round(profile['referenceFraction'] * (len(reference) - 1))
    chest = names['Chest']; count = len(original)
    yaw = _heading(original[:, chest]) - _heading(reference[frame, chest])
    align = Rotation.from_euler('y', yaw[:, None]).as_matrix()
    preserve = np.zeros(count, dtype=bool)
    if profile['stanceMode'] == 'crouching':
        preserve[:] = True
    elif profile['stanceMode'] in ('transition', 'mixed'):
        segment = points[:, names['Neck1']] - points[:, names['Spine1']]
        forward = np.stack((np.sin(_heading(original[:, chest])), np.zeros(count), np.cos(_heading(original[:, chest]))), axis=-1)
        incline = np.degrees(np.arctan2(np.sum(segment * forward, axis=-1), segment[:, 1]))
        preserve = incline >= 20.
    local = source['local_rot_mats'].copy()
    global_out = np.empty_like(original)
    offsets = np.asarray(rig['offsets'])
    phases = {}
    if profile['forearmSwing']:
        # The first harmonic of each leg's actual fore/aft path provides its
        # phase. Hands travel opposite their own foot, including lateral gaits.
        phase_time = np.arange(count) / max(1, count - 1)
        theta = 2 * np.pi * phase_time
        # Zero endpoint angular acceleration avoids a visible jerk and satisfies
        # the existing whole-skeleton seam-velocity gate at short Run periods.
        smooth_theta = 2 * np.pi * phase_time - np.sin(theta)
        for side, default in (('Left', 0.), ('Right', np.pi)):
            foot = points[:, names[side + 'Foot']] - points[:, 0]
            foot = np.einsum('tji,tj->ti', original[:, 0], foot)
            signal = foot[:, 2] - np.mean(foot[:-1, 2])
            harmonic = np.sum(signal[:-1] * np.exp(-1j * theta[:-1]))
            phase = float(np.angle(harmonic)) if abs(harmonic) > 1e-6 else default
            phases[side] = -np.cos(smooth_theta + phase)
    for i, parent in enumerate(rig['parents']):
        name = rig['jointNames'][i]
        if name in SPINE + NECK:
            baseline = align @ reference[frame, i]
            delta = baseline.swapaxes(-1, -2) @ original[:, i]
            gain = .08 if name in NECK else profile['bodyMotionGain']
            desired = baseline @ Rotation.from_rotvec(Rotation.from_matrix(delta).as_rotvec() * gain).as_matrix()
            authored = desired if parent < 0 else global_out[:, parent].swapaxes(-1, -2) @ desired
            local[:, i] = np.where(preserve[:, None, None], local[:, i], authored) if name in SPINE else authored
        elif name in ARMS:
            side = 'Left' if name.startswith('Left') else 'Right'
            if name.endswith(('Shoulder', 'Arm')) and not name.endswith('ForeArm'):
                # Fixed relative to the *current* chest, so leaning/crouching
                # moves the shoulder chain without introducing humeral swing.
                desired = global_out[:, chest] @ (reference[frame, chest].T @ reference[frame, i])
                local[:, i] = global_out[:, parent].swapaxes(-1, -2) @ desired
            elif name.endswith('ForeArm'):
                base = idle['local_rot_mats'][frame, i]
                if profile['forearmSwing']:
                    angle = np.radians(profile['elbowRestDegrees'] + profile['forearmSwingDegrees'] * phases[side])
                    target = np.stack((np.full(count, .025 if side == 'Left' else -.025), -np.cos(angle), np.sin(angle)), axis=-1)
                    target /= np.linalg.norm(target, axis=-1, keepdims=True)
                    target = np.einsum('tij,tj->ti', global_out[:, chest], target)
                    before = np.einsum('tij,j->ti', global_out[:, parent] @ base, offsets[names[side + 'Hand']])
                    before /= np.linalg.norm(before, axis=-1, keepdims=True)
                    axis = np.cross(before, target); sine = np.linalg.norm(axis, axis=-1)
                    axis /= np.maximum(sine[:, None], 1e-12)
                    rotation = Rotation.from_rotvec(axis * np.arctan2(sine, np.sum(before * target, axis=-1))[:, None]).as_matrix()
                    desired = rotation @ global_out[:, parent] @ base
                    local[:, i] = global_out[:, parent].swapaxes(-1, -2) @ desired
                else:
                    # Transient actions retain their authored elbow changes.
                    # Their arm chain follows the chest, rather than being
                    # attenuated wholesale together with the upper arms.
                    delta = base.T @ source['local_rot_mats'][:, i]
                    local[:, i] = base @ Rotation.from_rotvec(Rotation.from_matrix(delta).as_rotvec() * profile['forearmMotionGain']).as_matrix()
            else:
                local[:, i] = source['local_rot_mats'][:, i]
        global_out[:, i] = local[:, i] if parent < 0 else global_out[:, parent] @ local[:, i]
    untouched = [i for i, n in enumerate(rig['jointNames']) if n not in SPINE + NECK + ARMS]
    if not np.array_equal(local[:, untouched], source['local_rot_mats'][:, untouched]):
        raise ValueError('Forearm authoring changed root or lower body')
    spine = [names[n] for n in SPINE]
    if not np.array_equal(local[preserve][:, spine], source['local_rot_mats'][preserve][:, spine]):
        raise ValueError('Crouch spine was straightened')
    result = complete(local, source['root_positions'].copy(), rig, float(source['fps']))
    report = dict(policy=POLICY, profile=profile, referenceFrame=frame,
        inputMotionSha256=_array_hash(source['local_rot_mats'], source['root_positions'], source['fps']),
        outputMotionSha256=_array_hash(result['local_rot_mats'], result['root_positions'], result['fps']),
        idleMotionSha256=_array_hash(idle['local_rot_mats'], idle['root_positions'], idle['fps']),
        preservedRootAndLowerBody=True, preservedCrouchSpine=True, preservedSpineFrames=np.flatnonzero(preserve).tolist(),
        modifiedJoints=list(SPINE + NECK + ARMS), before=measure_posture(source, rig), after=measure_posture(result, rig),
        scope='Offline upper-body reauthoring; generated leg motion and its contact gates remain authoritative')
    return result, report


def arm_motion(motion, rig):
    rotations, points = fk(motion['local_rot_mats'], motion['root_positions'], rig)
    names = {n: i for i, n in enumerate(rig['jointNames'])}
    chest = names['Chest']; result = {}
    for side in ('Left', 'Right'):
        arm = names[side + 'Arm']; hand = names[side + 'Hand']
        relative = rotations[:, chest].swapaxes(-1, -2) @ rotations[:, arm]
        change = Rotation.from_matrix(relative[0].T @ relative).magnitude()
        travel = np.einsum('tji,tj->ti', rotations[:, chest], points[:, hand] - points[:, arm])
        result[side] = dict(upperArmMotionDegrees=float(np.degrees(change).max()),
            handForwardTravelMeters=float(np.ptp(travel[:, 2])),
            handVerticalTravelMeters=float(np.ptp(travel[:, 1])),
            handTravelMeters=float(np.linalg.norm(np.ptp(travel, axis=0))))
    return result


def audit_posture(motion, idle, rig, profile):
    from .locomotion_style import measure_posture
    validate_profile(profile)
    reference, observed = measure_posture(idle, rig), measure_posture(motion, rig)
    # Abduction is relative to the torso. A crouched chest's roll/lean must not
    # be reported as humeral spread in horizontal world coordinates.
    def abduction(value):
        rotations, points = fk(value['local_rot_mats'], value['root_positions'], rig)
        names = {n: i for i, n in enumerate(rig['jointNames'])}
        result = {}
        for side in ('Left', 'Right'):
            segment = points[:, names[side+'ForeArm']] - points[:, names[side+'Arm']]
            segment = np.einsum('tji,tj->ti', rotations[:, names['Chest']], segment)
            angles = np.degrees(np.arcsin(np.clip(np.abs(segment[:,0])/np.linalg.norm(segment,axis=-1),0.,1.)))
            result[side+'ArmAbductionDegrees']=dict(median=float(np.median(angles)),maximum=float(angles.max()))
        return result
    reference.update(abduction(idle)); observed.update(abduction(motion))
    arms = arm_motion(motion, rig); checks = []
    for key, value in observed.items():
        if key == 'torsoForwardDegrees' and profile['stanceMode'] != 'standing':
            continue  # User explicitly requires the crouched back to lean.
        limit_key = 'armAbductionDegrees' if 'ArmAbduction' in key else key
        for statistic in ('median', 'maximum'):
            limit = reference[key][statistic] + profile['limits'][limit_key]
            checks.append(dict(metric=key, statistic=statistic, observed=value[statistic], limit=limit, passed=value[statistic] <= limit))
    for side, values in arms.items():
        limit = profile['limits']['upperArmMotionDegrees']
        checks.append(dict(metric=side + 'UpperArmMotionDegrees', observed=values['upperArmMotionDegrees'], limit=limit,
                           passed=values['upperArmMotionDegrees'] <= limit))
        if profile['forearmSwing']:
            minimum = profile['limits']['minimumHandTravelMeters']
            checks.append(dict(metric=side + 'HandTravelMeters', observed=values['handTravelMeters'], minimum=minimum,
                               passed=values['handTravelMeters'] >= minimum))
    return dict(policy=POLICY, passed=all(c['passed'] for c in checks), checks=checks, profile=profile,
        idle=reference, observed=observed, arms=arms,
        motionSha256=_array_hash(motion['local_rot_mats'], motion['root_positions'], motion['fps']),
        idleMotionSha256=_array_hash(idle['local_rot_mats'], idle['root_positions'], idle['fps']),
        scope='Upper arms relative to chest and hand travel; crouch spine preservation is enforced by authoring; visual acceptance is separate')

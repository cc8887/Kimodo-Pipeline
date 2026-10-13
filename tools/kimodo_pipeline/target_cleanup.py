"""Bounded target-rig stance cleanup after native retargeting, before ALS gates."""
import copy
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from .gait_constraints import solve_foot_ik, _rotation_between
from .motion import fk


class TargetContactRepairError(ValueError):
    """Keep successful phase evidence when bounded IK is the failing stage."""
    def __init__(self, error, phase):
        super().__init__(str(error))
        self.phase = phase


def _bounded_orientation(original, desired):
    delta = Rotation.from_matrix(desired @ original.T).as_rotvec()
    size = np.linalg.norm(delta)
    delta *= min(1., np.deg2rad(19.9999)/max(size, 1e-12))
    return Rotation.from_rotvec(delta).as_matrix() @ original


def _bounded_hip_shift(hips, targets, reaches, limit):
    """Nearest shared pelvis displacement inside both fixed leg reach balls."""
    from scipy.optimize import minimize
    relative = np.asarray(hips)-np.asarray(targets)
    reaches = np.asarray(reaches)
    def feasible(delta):
        return reaches**2-np.sum((relative+delta)**2,axis=1)
    result=minimize(lambda d: float(d@d), np.zeros(3), jac=lambda d: 2*d,
        constraints=[dict(type='ineq',fun=feasible)], bounds=[(None,None),(None,None),(None,0.)],
        method='SLSQP',options=dict(maxiter=100,ftol=1e-10))
    delta=result.x
    if (not result.success or not np.isfinite(delta).all() or np.min(feasible(delta)) < -1e-6
            or np.linalg.norm(delta)>limit):
        raise ValueError(f'No reachable pelvis displacement inside the original hip budget: {np.linalg.norm(delta):.4f} cm')
    return delta


def cleanup_walk(raw, skeleton, source_gate, *, phase_shift=1, max_ankle_cm=8., max_hip_cm=5.):
    if not source_gate.get('pass') or source_gate['outputFrameCount'] != raw['sampledKeyCount']:
        raise ValueError('Target cleanup needs a passing source gate in the same time domain')
    observed = np.asarray(source_gate['transformedDetection']['sampleContacts'], bool).all(axis=-1)
    result, report = _cleanup_contacts(raw, skeleton, observed, 150., (0., 1.),
        phase_shift=phase_shift, max_ankle_cm=max_ankle_cm, max_hip_cm=max_hip_cm)
    report['sourceGatePolicy'] = source_gate['policy']['version']
    return result, report


def repair_periodic_gait(raw, skeleton, source, *, phase_harmonics=3, source_gate=None):
    from .target_phase import infer_phases, infer_verified_source_phases, extend_trailing_low_support
    from .target_gait import PILOTS, DIRECTIONS, asset_name, original_protocol, analyze_target_gait
    name = asset_name(source)
    speed, kind = PILOTS[name]
    contact_mode=original_protocol()['assets'][kind]['contactMode']
    try:
        phase = infer_phases(raw, skeleton, source,contact_mode=contact_mode, harmonics=phase_harmonics)
    except ValueError as error:
        if source_gate is None or analyze_target_gait(raw, skeleton, source)['passed']:
            raise
        try:
            phase = infer_verified_source_phases(raw, skeleton, source, source_gate, contact_mode=contact_mode)
        except ValueError as fallback_error:
            raise ValueError(str(error)+'; source FK fallback: '+str(fallback_error)) from fallback_error
        phase['verticalOnlyInferenceFailure'] = str(error)
    try:
        result, report = _cleanup_contacts(raw, skeleton, np.asarray(phase['candidateContacts'], bool),
            speed, DIRECTIONS[name], phase_shift=0,contact_mode=contact_mode)
    except ValueError as error:
        raise TargetContactRepairError(error, phase) from error
    if contact_mode == 'toe':
        gate = analyze_target_gait(result, skeleton, source)
        if not gate['passed']:
            attempts = [dict(pinForefootOrientation=False, passed=False,
                             failedChecks=[c for c in gate['checks'] if not c['passed']])]
            original_intent = np.asarray(phase['candidateContacts'], bool)
            best = None
            for padding in (0, 1):
                intent = (original_intent if padding == 0 else
                          extend_trailing_low_support(raw, skeleton, source, original_intent))
                if padding and np.array_equal(intent, original_intent):
                    continue
                for gain in (.05, .1, .2, .25, 1.):
                    attempt = dict(pinForefootOrientation=True, trailingLowWindowFrames=padding,
                                   forefootOrientationGain=gain)
                    try:
                        alternative, alternative_report = _cleanup_contacts(raw, skeleton,
                            intent, speed, DIRECTIONS[name], phase_shift=0, contact_mode=contact_mode,
                            pin_forefoot_orientation=True, forefoot_orientation_gain=gain,
                            allow_planar_hip=True)
                        alternative_gate = analyze_target_gait(alternative, skeleton, source)
                        attempt.update(passed=alternative_gate['passed'],
                            failedChecks=[c for c in alternative_gate['checks'] if not c['passed']])
                        if alternative_gate['passed']:
                            score=max(c['medianCmS']/c['limitCmS'] for c in alternative_gate['checks']
                                      if 'independent_near_ground_sliding' in c['check'])
                            attempt['maximumSlidingLimitFraction']=score
                            if best is None or score<best[0]:
                                best=(score,alternative,alternative_report,intent.copy(),padding,len(attempts))
                    except ValueError as error:
                        attempt.update(passed=False, error=str(error))
                    attempts.append(attempt)
            if best is not None:
                _,result,report,intent,padding,index=best
                attempts[index]['selected']=True
                phase = copy.deepcopy(phase)
                phase['unexpandedCandidateContacts'] = phase['candidateContacts']
                phase['candidateContacts'] = intent.astype(int).tolist()
                phase['trailingLowWindowPaddingFrames'] = padding
                phase['extensionHeightAllowanceCm'] = 1. if padding else 0.
            report['forefootRepairCandidates'] = attempts
    report.update(phasePolicy=phase['policy'], passed=True, applied=True)
    return result, phase, report


def _cleanup_contacts(raw, skeleton, observed, speed, direction, *, phase_shift=0,
                      max_ankle_cm=8., max_hip_cm=5.,contact_mode='flat_sole',pin_forefoot_orientation=False,
                      forefoot_orientation_gain=1.,allow_planar_hip=False):
    from .publish import world_pose
    result = copy.deepcopy(raw); count = raw['sampledKeyCount']; period = count-1
    if contact_mode not in ('flat_sole','toe'):
        raise ValueError('Unsupported target contact mode')
    if not np.isfinite(forefoot_orientation_gain) or not 0 < forefoot_orientation_gain <= 1:
        raise ValueError('Invalid forefoot orientation interpolation gain')
    if observed.shape != (count, 2):
        raise ValueError('Candidate phase contact dimensions differ')
    indices = (np.arange(count)-phase_shift) % period if phase_shift else np.arange(count)
    if phase_shift:
        for track in result['tracks']:
            for key in ('positions', 'rotations', 'scales'):
                if len(track[key]) == count:
                    track[key] = np.asarray(track[key])[indices].tolist()
    names = skeleton['rawBoneNames']; lookup = {n.lower(): i for i, n in enumerate(names)}
    tracks = {t['bone'].lower(): t for t in result['tracks']}
    if any(t['scales'] and not np.allclose(t['scales'], 1, atol=1e-6) for t in result['tracks']):
        raise ValueError('Target stance cleanup requires unit bone scales')
    world = world_pose(result, skeleton)
    bind = world_pose(dict(sampledKeyCount=1, tracks=[]), skeleton)
    local = np.stack([Rotation.from_quat(np.broadcast_to(
        np.asarray(tracks[n.lower()]['rotations']), (count, 4))).as_matrix() for n in names], axis=1)
    positions = np.stack([np.broadcast_to(np.asarray(tracks[n.lower()]['positions']), (count, 3)) for n in names], axis=1).copy()
    observed = observed[indices]
    target = np.stack([world['foot_'+s][1] for s in ('l', 'r')], axis=1).copy()
    target_rot = np.stack([world['foot_'+s][0] for s in ('l', 'r')], axis=1).copy()
    toe_rot = np.stack([local[:, lookup['ball_'+s]] for s in ('l', 'r')], axis=1).copy()
    maximum_roll_correction=0.; maximum_orientation_correction=0.
    for side, suffix in enumerate(('l', 'r')):
        foot, ball = 'foot_'+suffix, 'ball_'+suffix
        f_idx, b_idx = lookup[foot], lookup[ball]
        heel_offset = bind[foot][0][0].T @ np.array([0., 0., -bind[foot][1][0, 2]])
        toe_offset = bind[ball][0][0].T @ np.array([0., 0., -bind[ball][1][0, 2]])
        flat = observed[:, side]
        edges = np.diff(np.r_[False, flat[:-1], False].astype(int))
        intervals = list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))
        if len(intervals) > 1 and intervals[0][0] == 0 and intervals[-1][1] == period:
            intervals = [(intervals[-1][0], intervals[0][1]+period), *intervals[1:-1]]
        for first, stop in intervals:
            if stop-first < 3:
                continue
            middle = ((first+stop-1)//2) % period
            forward = world[ball][1][middle]-world[foot][1][middle]
            rest_forward = bind[ball][1][0]-bind[foot][1][0]
            yaw = np.arctan2(forward[1], forward[0])-np.arctan2(rest_forward[1], rest_forward[0])
            rotation = Rotation.from_euler('z', yaw).as_matrix() @ bind[foot][0][0]
            span = np.arange(first, stop); wrapped = span % period
            advancing = np.zeros((len(span), 3))
            advancing[:, :2] = span[:, None]*(raw['playLength']/period)*speed*np.asarray(direction)
            heels = world[foot][1][wrapped]+np.einsum('tij,j->ti', world[foot][0][wrapped], heel_offset)+advancing
            samples=(world[ball][1][wrapped]+np.einsum('tij,j->ti',world[ball][0][wrapped],toe_offset)+advancing
                     if contact_mode=='toe' else heels)
            anchor = np.median(samples, axis=0); anchor[2] = 0
            ankle = anchor-rotation@heel_offset
            for unwrapped in range(first-2, stop+2):
                frame = unwrapped % period
                alpha = min(1., (unwrapped-first+2)/2, (stop+1-unwrapped)/2)
                alpha = np.clip(alpha, 0, 1); alpha = alpha*alpha*(3-2*alpha)
                advancing = np.zeros(3); advancing[:2] = unwrapped*(raw['playLength']/period)*speed*np.asarray(direction)
                if contact_mode=='toe':
                    desired=world[foot][0][frame]
                    pin_this_frame=pin_forefoot_orientation and first <= unwrapped < stop
                    if pin_this_frame:
                        desired=Slerp([0,1], Rotation.from_matrix(np.stack([
                            world[foot][0][frame],world[foot][0][middle]])))([forefoot_orientation_gain]).as_matrix()[0]
                        toe_rot[frame,side]=Slerp([0,1], Rotation.from_matrix(np.stack([
                            local[frame,b_idx], local[middle,b_idx]])))([alpha*forefoot_orientation_gain]).as_matrix()[0]
                        toe_rot[frame,side]=_bounded_orientation(local[frame,b_idx],toe_rot[frame,side])
                    # Lift only a sinking heel, rotating around the pinned toe.
                    toe_vector=positions[frame,b_idx]+toe_rot[frame,side]@toe_offset
                    heel_from_toe=desired@(heel_offset-toe_vector)
                    if heel_from_toe[2]<0:
                        horizontal=heel_from_toe.copy();horizontal[2]=0
                        if np.linalg.norm(horizontal)<1e-6:raise ValueError('Unresolved forefoot pitch')
                        pitch=_rotation_between(heel_from_toe,horizontal)
                        angle=float(np.rad2deg(np.linalg.norm(Rotation.from_matrix(pitch).as_rotvec())))
                        if angle>20.:raise ValueError('Target forefoot pitch correction exceeds 20 degrees')
                        maximum_roll_correction=max(maximum_roll_correction,angle*alpha)
                        desired=pitch@desired
                    if pin_this_frame:
                        desired=_bounded_orientation(world[foot][0][frame],desired)
                    target_rot[frame,side]=Slerp([0,1],Rotation.from_matrix(np.stack([world[foot][0][frame],desired])))([alpha]).as_matrix()[0]
                    if pin_this_frame:
                        changes = Rotation.from_matrix(np.stack([
                            target_rot[frame,side]@world[foot][0][frame].T,
                            toe_rot[frame,side]@local[frame,b_idx].T])).magnitude()
                        angle=float(np.rad2deg(changes.max()))
                        if angle>20.:raise ValueError(f'Pinned forefoot orientation exceeds 20 degrees: {suffix}/{frame}: {angle:.6f}')
                        maximum_orientation_correction=max(maximum_orientation_correction,angle)
                    ball_vector=target_rot[frame,side]@toe_vector
                    ankle=anchor-ball_vector
                target[frame, side] = world[foot][1][frame]*(1-alpha)+(ankle-advancing)*alpha
                if contact_mode=='flat_sole':
                    target_rot[frame, side] = Slerp([0, 1], Rotation.from_matrix(np.stack([world[foot][0][frame], rotation])))([alpha]).as_matrix()[0]
                    reference = Rotation.from_quat(skeleton['referencePose'][b_idx]['rotation']).as_matrix()
                    toe_rot[frame, side] = Slerp([0, 1], Rotation.from_matrix(np.stack([local[frame, b_idx], reference])))([alpha]).as_matrix()[0]
        # Preserve the repeated pose endpoint; the reconstructed world travel
        # still advances one complete stride across the seam.
        target[-1, side] = target[0, side]
        target_rot[-1, side] = target_rot[0, side]
        toe_rot[-1, side] = toe_rot[0, side]
        for frame in range(count):
            r = target_rot[frame, side]
            heel_z = (r @ heel_offset)[2]
            toe_z = (r @ positions[frame, b_idx]+r @ toe_rot[frame, side] @ toe_offset)[2]
            target[frame, side, 2] = max(target[frame, side, 2], -min(heel_z, toe_z))
    original_ankles = np.stack([world['foot_'+s][1] for s in ('l', 'r')], axis=1)
    correction = float(np.max(np.linalg.norm(target-original_ankles, axis=-1)))
    if correction > max_ankle_cm:
        raise ValueError(f'Target foot correction exceeds bound: {correction:.4f} cm')
    aliases = {'pelvis': 'Hips', 'thigh_l': 'LeftLeg', 'calf_l': 'LeftShin', 'foot_l': 'LeftFoot', 'ball_l': 'LeftToeEnd',
               'thigh_r': 'RightLeg', 'calf_r': 'RightShin', 'foot_r': 'RightFoot', 'ball_r': 'RightToeEnd'}
    rig = dict(jointNames=[aliases.get(n.lower(), n) for n in names], parents=skeleton['rawParents'])
    changes = []; hip_displacements=[]
    pelvis = lookup['pelvis']; parent = rig['parents'][pelvis]
    for frame in range(count):
        lower = 0.; hips=[]; reaches=[]
        for side, suffix in enumerate(('l', 'r')):
            hip, knee, ankle = [world[n+'_'+suffix][1][frame] for n in ('thigh', 'calf', 'foot')]
            reach = np.linalg.norm(knee-hip)+np.linalg.norm(ankle-knee)-1e-3
            hips.append(hip);reaches.append(reach)
            horizontal2 = np.sum((hip[:2]-target[frame, side, :2])**2)
            if horizontal2 >= reach*reach:
                raise ValueError('Target stance has unreachable horizontal foot placement')
            lower = max(lower, hip[2]-target[frame, side, 2]-np.sqrt(reach*reach-horizontal2))
        displacement=np.array([0.,0.,-lower])
        if lower > max_hip_cm:
            if not allow_planar_hip:
                raise ValueError(f'Target hip correction exceeds bound: {lower:.4f} cm')
            displacement=_bounded_hip_shift(hips,target[frame],reaches,max_hip_cm)
        positions[frame, pelvis] += world[names[parent].lower()][0][frame].T @ displacement
        hip_displacements.append(float(np.linalg.norm(displacement)))
        rig['offsets'] = positions[frame].tolist()
        for side, label in enumerate(('left', 'right')):
            local[frame] = solve_foot_ik(local[frame], positions[frame, 0], rig, label,
                target[frame, side], target_rot[frame, side], {lookup['ball_'+('l','r')[side]]: toe_rot[frame, side]})
        changes.append(float(-displacement[2]))
    for index, name in enumerate(names):
        if name.lower() in ('thigh_l','calf_l','foot_l','ball_l','thigh_r','calf_r','foot_r','ball_r'):
            tracks[name.lower()]['rotations'] = Rotation.from_matrix(local[:, index]).as_quat().tolist()
        if index == pelvis:
            tracks[name.lower()]['positions'] = positions[:, index].tolist()
    report = dict(method='target_world_stance_ik_v1', phaseShiftFrames=phase_shift,
                  contactMode=contact_mode,
                  pinForefootOrientation=pin_forefoot_orientation,
                  forefootOrientationGain=forefoot_orientation_gain if pin_forefoot_orientation else 0.,
                  maximumForefootOrientationCorrectionDegrees=maximum_orientation_correction,
                  maximumFootRollCorrectionDegrees=maximum_roll_correction,
                  maximumAnkleCorrectionCm=correction, maximumHipLoweringCm=max(changes),
                  maximumHipCorrectionCm=max(hip_displacements), allowPlanarHip=allow_planar_hip,
                  expectedSpeedCmS=speed, travelDirectionXY=list(direction), qualityGateRequired=True)
    return result, report

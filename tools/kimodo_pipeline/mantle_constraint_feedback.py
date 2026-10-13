"""Project failed candidate poses into task feasibility for sparse generation retries.

This is a bounded root/torso projection with two-link contact IK, not a general
physics solver or an asset repair certificate. Only regenerated motion can pass
the existing author/target/runtime gates. The reference supplies task anchors
and the accepted root tube; it never supplies the replacement full-body pose.
"""
from __future__ import annotations

import copy
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from .motion import fk, resample, validate_motion
from .gait_constraints import solve_foot_ik
from .prompt_modules import fingerprint

POLICY = 'mantle-candidate-pose-feedback-v1'


class ProjectionError(ValueError):
    def __init__(self, message, receipt):
        self.receipt = receipt
        super().__init__(message)


def _limb(rig, contact):
    names = {name:index for index,name in enumerate(rig['jointNames'])}
    if contact['side'] not in ('left', 'right') or contact['kind'] not in ('hand', 'foot'):
        raise ValueError('Expected a left/right hand/foot contact')
    suffixes = ('Arm', 'ForeArm', 'Hand') if contact['kind'] == 'hand' else ('Leg', 'Shin', 'Foot')
    indices = [names[contact['side'].title()+suffix] for suffix in suffixes]
    if rig['parents'][indices[1]] != indices[0] or rig['parents'][indices[2]] != indices[1]:
        raise ValueError('Contact requires a direct two-link limb')
    lengths = [float(np.linalg.norm(rig['offsets'][index])) for index in indices[1:]]
    if min(lengths) <= 0:
        raise ValueError('Contact limb has zero length')
    return indices, lengths


def project_pose(local, root, rig, reference_root, contacts, *, root_tube_meters=.15,
                 joint_limit_degrees=40., reach_margin_meters=.005, projection_tube_meters=None):
    """Find a nearby feasible pose, keeping every nonparticipating local joint.

    The 0.9 factor reserves numerical/inter-frame margin inside the existing
    root gate; it does not change that gate. Hand/foot targets use fixed lengths.
    """
    local, root, reference_root = (np.asarray(value, dtype=float) for value in (local,root,reference_root))
    if (root.shape != (3,) or reference_root.shape != (3,) or
            not np.isfinite(root).all() or not np.isfinite(reference_root).all() or
            not np.isfinite(root_tube_meters) or not 0 < root_tube_meters <= .15 or
            not 0 < joint_limit_degrees <= 60 or not 0 < reach_margin_meters < .2):
        raise ValueError('Invalid bounded pose-projection inputs')
    validate_motion(dict(local_rot_mats=local[None], root_positions=root[None], fps=30), rig)
    contacts = copy.deepcopy(contacts)
    labels = [contact['kind']+':'+contact['side'] for contact in contacts]
    if len(set(labels)) != len(labels):
        raise ValueError('Repeated contact target')
    limbs = []
    for contact in contacts:
        contact['position'] = np.asarray(contact['position'],dtype=float)
        contact['rotation'] = np.asarray(contact['rotation'],dtype=float)
        if (contact['position'].shape != (3,) or contact['rotation'].shape != (3,3) or
                not np.isfinite(contact['position']).all() or not np.isfinite(contact['rotation']).all() or
                not np.allclose(contact['rotation']@contact['rotation'].T,np.eye(3),atol=1e-7) or
                not np.isclose(np.linalg.det(contact['rotation']),1.,atol=1e-7)):
            raise ValueError('Invalid contact position/orientation')
        limb=_limb(rig,contact)
        reserve=float(contact.get('reachMarginMeters',reach_margin_meters))
        if not np.isfinite(reserve) or not 0 < reserve < min(limb[1]):
            raise ValueError('Contact reach margin consumes the limb workspace')
        contact['reachMarginMeters']=reserve
        limbs.append(limb)
    names = {name:index for index,name in enumerate(rig['jointNames'])}
    hand_sides = {contact['side'] for contact in contacts if contact['kind']=='hand'}
    adjustable = ['Hips','Spine1','Spine2','Chest'] if hand_sides else []
    adjustable += [side.title()+'Shoulder' for side in sorted(hand_sides)]
    adjustable = [names[name] for name in adjustable if name in names]
    dimension = 3+3*len(adjustable)
    angular_limit = np.deg2rad(joint_limit_degrees)
    tube = root_tube_meters*.9 if projection_tube_meters is None else float(projection_tube_meters)
    if not np.isfinite(tube) or not 0 < tube <= root_tube_meters:
        raise ValueError('Projection tube must remain inside the unchanged root gate')

    def pose(parameters):
        rotations = local.copy()
        for index,delta in zip(adjustable,parameters[3:].reshape(-1,3)):
            rotations[index] = local[index]@Rotation.from_rotvec(delta).as_matrix()
        return rotations, root+parameters[:3]

    def constraints(parameters):
        rotations,position=pose(parameters)
        _,points=fk(rotations[None],position[None],rig)
        inequalities=[tube**2-float(np.sum((position-reference_root)**2))]
        for contact,(indices,lengths) in zip(contacts,limbs):
            distance2=float(np.sum((points[0,indices[0]]-contact['position'])**2))
            outer=sum(lengths)-contact['reachMarginMeters']
            inner=abs(lengths[0]-lengths[1])+contact['reachMarginMeters']
            inequalities.extend([outer**2-distance2,distance2-inner**2])
        for delta in parameters[3:].reshape(-1,3):
            inequalities.append(angular_limit**2-float(delta@delta))
        return np.asarray(inequalities)

    def objective(parameters):
        return float(np.sum((parameters[:3]/.08)**2)+np.sum((parameters[3:]/.35)**2))

    initial=np.zeros(dimension)
    delta=root-reference_root
    if np.linalg.norm(delta)>tube:
        initial[:3]=reference_root+delta/np.linalg.norm(delta)*tube-root
    fitted=minimize(objective,initial,method='SLSQP',constraints=[dict(type='ineq',fun=constraints)],
        bounds=[(None,None)]*3+[(-angular_limit,angular_limit)]*(dimension-3),
        options=dict(maxiter=240,ftol=1e-11))
    receipt=dict(schemaVersion=1,policy=POLICY,scope='conditioning_pose_only',deliverable=False,
        solverSuccess=bool(fitted.success),solverMessage=str(fitted.message),
        originalRootMeters=root.tolist(),referenceRootMeters=reference_root.tolist(),
        rootGateMeters=root_tube_meters,projectionTubeMeters=tube,
        maximumJointDeltaDegrees=joint_limit_degrees,contactKinds=labels,
        minimumConstraintValue=float(np.min(constraints(fitted.x))),passed=False)
    if not fitted.success or np.min(constraints(fitted.x)) < -1e-8:
        raise ProjectionError('No feasible bounded candidate-pose projection',receipt)
    projected, projected_root=pose(fitted.x)
    from .mantle import _arm_rig
    try:
        for contact in contacts:
            contact_rig=_arm_rig(rig) if contact['kind']=='hand' else rig
            projected=solve_foot_ik(projected,projected_root,contact_rig,contact['side'],
                                    contact['position'],contact['rotation'],{})
    except ValueError as error:
        receipt['ikError']=str(error)
        raise ProjectionError('Projected contact IK failed',receipt) from error
    _,before=fk(local[None],root[None],rig)
    after_rot,after=fk(projected[None],projected_root[None],rig)
    angular_changes=np.rad2deg(Rotation.from_matrix(local.swapaxes(-1,-2)@projected).magnitude())
    receipt.update(projectedRootMeters=projected_root.tolist(),
        rootCorrectionCm=float(np.linalg.norm(projected_root-root)*100),
        rootErrorBeforeCm=float(np.linalg.norm(root-reference_root)*100),
        rootErrorAfterCm=float(np.linalg.norm(projected_root-reference_root)*100),
        affectedJoints=[dict(name=name,localRotationChangeDegrees=float(angular_changes[index]))
            for name,index in names.items() if angular_changes[index]>1e-7],
        contacts=[])
    for contact,(indices,lengths) in zip(contacts,limbs):
        end=indices[-1]
        receipt['contacts'].append(dict(kind=contact['kind'],side=contact['side'],
            armOrLegLengthCm=sum(lengths)*100,
            projectedReachReserveCm=contact['reachMarginMeters']*100,
            beforePositionErrorCm=float(np.linalg.norm(before[0,end]-contact['position'])*100),
            afterPositionErrorCm=float(np.linalg.norm(after[0,end]-contact['position'])*100),
            afterOrientationErrorDegrees=float(np.rad2deg(Rotation.from_matrix(
                contact['rotation'].T@after_rot[0,end]).magnitude()))))
    receipt['passed']=(receipt['rootErrorAfterCm']<=root_tube_meters*100+1e-5 and
        all(row['afterPositionErrorCm']<1e-4 and row['afterOrientationErrorDegrees']<1e-4 for row in receipt['contacts']))
    if not receipt['passed']:
        raise ProjectionError('Projected pose failed independent reconstruction',receipt)
    return projected,projected_root,receipt


def _events(spec,count,duration):
    from .mantle import _hand_frames
    hands={side:set(_hand_frames(spec,side,count,duration).tolist()) for side in spec['hands']}
    feet=set(np.flatnonzero(np.linspace(0,duration,count)>=spec['groundedSeconds'][1]-1e-7).tolist())
    return hands,feet


def select_failure_keys(candidate,reference,rig,spec,duration,*,limit=4,excluded=()):
    """Select extrema from actual task violations, never asset-specific frame numbers."""
    count=len(candidate['root_positions'])
    if count!=len(reference['root_positions']):
        raise ValueError('Feedback motions must share a timeline')
    ar,ap=fk(candidate['local_rot_mats'],candidate['root_positions'],rig)
    gr,gp=fk(reference['local_rot_mats'],reference['root_positions'],rig)
    names={name:index for index,name in enumerate(rig['jointNames'])}
    hands,feet=_events(spec,count,duration)
    metrics=[('pelvis-tube',np.linalg.norm(candidate['root_positions']-reference['root_positions'],axis=1)/.15,list(range(count)))]
    for side,frames in hands.items():
        index=names[side.title()+'Hand']
        metrics.append((side+'-hand-repair-budget',np.linalg.norm(ap[:,index]-gp[:,index],axis=1)/.08,sorted(frames)))
        angles=np.rad2deg(Rotation.from_matrix(gr[:,index].swapaxes(-1,-2)@ar[:,index]).magnitude())
        metrics.append((side+'-palm-orientation',angles/12.,sorted(frames)))
        shoulder=names[side.title()+'Arm']
        length=sum(np.linalg.norm(rig['offsets'][names[side.title()+part]]) for part in ('ForeArm','Hand'))
        metrics.append((side+'-shoulder-reach',np.linalg.norm(ap[:,shoulder]-gp[:,index],axis=1)/length,sorted(frames)))
    for side in ('left','right'):
        index=names[side.title()+'Foot']
        metrics.append((side+'-landing-foot',np.linalg.norm(ap[:,index]-gp[:,index],axis=1)/.03,sorted(feet)))
        angles=np.rad2deg(Rotation.from_matrix(gr[:,index].swapaxes(-1,-2)@ar[:,index]).magnitude())
        metrics.append((side+'-landing-foot-orientation',angles/12.,sorted(feet)))
    selected={}
    for reason,ratios,frames in metrics:
        active=[frame for frame in frames if frame not in excluded and ratios[frame]>1+1e-6]
        if not active:
            continue
        # Every disjoint failure interval gets its own measured extremum;
        # one global wrist-error peak can conceal a different unreachable arm.
        runs=[]
        for frame in active:
            if not runs or frame!=runs[-1][-1]+1:
                runs.append([])
            runs[-1].append(frame)
        for run in runs:
            frame=max(run,key=lambda i:ratios[i])
            selected.setdefault(frame,[]).append(dict(metric=reason,ratio=float(ratios[frame]),
                intervalFrames=[run[0],run[-1]]))
    ranking=sorted(selected,key=lambda frame:(-max(item['ratio'] for item in selected[frame]),frame))[:limit]
    return [dict(frame=frame,seconds=float(frame/(count-1)*duration),failures=selected[frame]) for frame in sorted(ranking)]


def build_feedback_constraints(params,candidate,reference,rig,conditions,*,maximum_added_frames=4,adaptive_margins=False,
                               reuse_existing_time_keys=False):
    """Append projected fullbody keys and reconcile overlapping EE/root conditions.

    Existing fullbody keys are preserved. Per-type and total key unions stay
    below 20. Caller must keep all original acceptance and retry provenance.
    """
    from .mantle import profile,reference_input
    native,_,identity=reference_input(params['semantic'])
    count=int(params['seconds']*params['fps'])
    candidate=resample(candidate,count,rig)
    reference=resample(reference,count,rig)
    validate_motion(candidate,rig);validate_motion(reference,rig)
    conditions=copy.deepcopy(conditions)
    spec=profile(params['semantic']);duration=native['playLength']
    fullbody=next((condition for condition in conditions if condition['type']=='fullbody'),None)
    if fullbody is None:
        raise ValueError('Sparse feedback requires explicit entry/exit fullbody conditions')
    selected=select_failure_keys(candidate,reference,rig,spec,duration,
        limit=maximum_added_frames,excluded=set(fullbody['frame_indices']))
    if not selected:
        raise ProjectionError('No new bounded feedback keys selected',dict(policy=POLICY,passed=False,reason='no remaining eligible extrema'))
    occupied={frame for condition in conditions for frame in condition['frame_indices']}
    hands,feet=_events(spec,count,duration)
    if reuse_existing_time_keys:
        # Upgrade the nearest soft event key in the same support phase. This
        # places an exact projected pose close to an unpinned error extremum
        # while retaining the original total temporal sampling budget.
        def phase(frame):
            return (tuple(side for side,frames in hands.items() if frame in frames),frame in feet)
        upgraded={}
        for row in selected:
            peak=row['frame']
            eligible=[frame for frame in occupied if frame not in fullbody['frame_indices'] and phase(frame)==phase(peak)]
            if not eligible:
                continue
            frame=min(eligible,key=lambda value:(abs(value-peak),value))
            upgraded.setdefault(frame,dict(frame=frame,seconds=float(frame/(count-1)*duration),failures=[],
                measuredFailureFrames=[],selection='nearest existing event key in the same support phase'))
            upgraded[frame]['failures'].extend(row['failures'])
            upgraded[frame]['measuredFailureFrames'].append(peak)
        selected=[upgraded[frame] for frame in sorted(upgraded)]
    allowed=[]
    for row in selected:
        if row['frame'] in occupied or len(occupied)<19:
            allowed.append(row);occupied.add(row['frame'])
    if not allowed:
        raise ProjectionError('Feedback sparse-key budget exhausted',dict(policy=POLICY,passed=False))
    gr,gp=fk(reference['local_rot_mats'],reference['root_positions'],rig)
    _,ap=fk(candidate['local_rot_mats'],candidate['root_positions'],rig)
    names={name:index for index,name in enumerate(rig['jointNames'])}
    root_errors=np.linalg.norm(candidate['root_positions']-reference['root_positions'],axis=1)
    key_frames=sorted(fullbody['frame_indices'])
    overshoots=[max(0.,float(np.max(root_errors[start:end+1])-max(root_errors[start],root_errors[end])))
                for start,end in zip(key_frames,key_frames[1:])]
    observed_overshoot=max(overshoots,default=0.)
    projection_tube=min(.135,.15-observed_overshoot-.005) if adaptive_margins else .135
    if projection_tube<=0:
        raise ProjectionError('Measured root overshoot consumes the accepted tube; add semantic events or reject',
            dict(policy=POLICY,passed=False,observedRootOvershootCm=observed_overshoot*100))
    reports=[]
    for row in allowed:
        frame=row['frame'];contacts=[]
        for side,frames in hands.items():
            if frame in frames:
                index=names[side.title()+'Hand']
                contact=dict(kind='hand',side=side,position=gp[frame,index],rotation=gr[frame,index])
                if adaptive_margins:
                    shoulder=names[side.title()+'Arm']
                    length=sum(np.linalg.norm(rig['offsets'][names[side.title()+part]]) for part in ('ForeArm','Hand'))
                    interval=sorted(frames)
                    unpinned=[i for i in interval if i not in key_frames]
                    excess=max([max(0.,float(np.linalg.norm(ap[i,shoulder]-gp[i,index])-length)) for i in unpinned],default=0.)
                    steps=[float(np.linalg.norm((ap[i,shoulder]-gp[i,index])-(ap[i-1,shoulder]-gp[i-1,index])))
                           for i in interval if i-1 in frames]
                    motion_reserve=max(steps,default=0.)
                    contact['reachMarginMeters']=max(.005,excess+motion_reserve)
                    contact['marginEvidence']=dict(unpinnedMaximumExcessCm=excess*100,
                        maximumAdjacentShoulderToAnchorMotionCm=motion_reserve*100,
                        derivedReserveCm=contact['reachMarginMeters']*100)
                contacts.append(contact)
        if frame in feet:
            for side in ('left','right'):
                index=names[side.title()+'Foot']
                contacts.append(dict(kind='foot',side=side,position=gp[frame,index],rotation=gr[frame,index]))
        projected,root,receipt=project_pose(candidate['local_rot_mats'][frame],candidate['root_positions'][frame],rig,
                                          reference['root_positions'][frame],contacts,projection_tube_meters=projection_tube)
        receipt['adaptiveMarginEvidence']=dict(enabled=adaptive_margins,
            rootInterpolationOvershootCm=observed_overshoot*100,projectionRootTubeCm=projection_tube*100,
            hands={contact['side']:contact['marginEvidence'] for contact in contacts if 'marginEvidence' in contact})
        rotations=Rotation.from_matrix(projected).as_rotvec().tolist()
        for condition in conditions:
            if frame in condition['frame_indices']:
                index=condition['frame_indices'].index(frame)
                if 'local_joints_rot' in condition:
                    condition['local_joints_rot'][index]=rotations
                    condition['root_positions'][index]=root.tolist()
                if 'smooth_root_2d' in condition:
                    condition['smooth_root_2d'][index]=root[[0,2]].tolist()
        fullbody['frame_indices'].append(frame)
        fullbody['local_joints_rot'].append(rotations)
        fullbody['root_positions'].append(root.tolist())
        reports.append(dict(**row,projection=receipt,
            originalLocalRotations=Rotation.from_matrix(candidate['local_rot_mats'][frame]).as_rotvec().tolist(),
            projectedLocalRotations=rotations))
    order=np.argsort(fullbody['frame_indices']).tolist()
    for key in ('frame_indices','local_joints_rot','root_positions'):
        fullbody[key]=[fullbody[key][index] for index in order]
    if any(len(condition['frame_indices'])>=20 for condition in conditions) or len(occupied)>=20:
        raise ValueError('Feedback constraints exceed sparse-key budget')
    receipt=dict(schemaVersion=1,policy=POLICY,reference=identity,scope='generation_conditions_only',
        deliverable=False,productionReady=False,passed=True,selected=reports,totalKeyUnion=sorted(occupied),
        fullBodyFrames=fullbody['frame_indices'],
        freeLocalJointsPreserved=True,referenceFullBodyPoseCopied=False,
        adaptiveMargins=adaptive_margins,
        reusedExistingTimeKeys=reuse_existing_time_keys,
        unchangedAuthorRepairBudgets=dict(handPositionCm=8.,handOrientationDegrees=30.),
        constraintsSha256=fingerprint(conditions))
    return conditions,receipt

"""Add candidate-derived sparse conditions for the unchanged target repair budget."""
from __future__ import annotations

import copy
import numpy as np
from scipy.spatial.transform import Rotation,Slerp

from .graph import digest
from .motion import fk,resample
from .mantle_constraint_feedback import _events
from .mantle_target_projection import TargetPoseObserver,TargetProjectionError,project_target_pose

POLICY='mantle-target-budget-feedback-v1'


def build_target_feedback_constraints(params,motion,source_reference,rig,conditions,*,maximum_frames=2):
    if type(maximum_frames) is not int or maximum_frames<1:
        raise ValueError('Target feedback requires a positive frame budget')
    from .mantle import reference_input,profile,TARGET_HAND_CORRECTION_CM
    from .publish import world_pose
    semantic=params['semantic'];native,skeleton,identity=reference_input(semantic)
    count=int(params['seconds']*params['fps'])
    motion=resample(motion,count,rig);source_reference=resample(source_reference,count,rig)
    spec=profile(semantic);hands,feet=_events(spec,count,native['playLength'])
    source_rot,source_points=fk(motion['local_rot_mats'],motion['root_positions'],rig)
    source_names={name:index for index,name in enumerate(rig['jointNames'])}
    native_world=world_pose(native,skeleton)
    times=np.linspace(0,1,count);native_times=np.linspace(0,1,native['sampledKeyCount'])
    target_points={};target_rot={}
    for side in hands:
        rotations,points,_=native_world['hand_'+side[0]]
        target_points[side]=np.stack([np.interp(times,native_times,points[:,axis]) for axis in range(3)],axis=1)
        target_points[side]+=np.asarray(spec.get('targetMeshHandOffsetsCm',{}).get(side,(0,0,0)))
        target_rot[side]=Slerp(native_times,Rotation.from_matrix(rotations))(times).as_matrix()
    observer=TargetPoseObserver(rig,skeleton)
    predicted=np.stack([observer.observe(local,root)[1] for local,root in zip(motion['local_rot_mats'],motion['root_positions'])])
    errors={side:np.linalg.norm(predicted[:,observer.lookup['hand_'+side[0]]]-target_points[side],axis=1) for side in hands}
    conditions=copy.deepcopy(conditions)
    occupied=sorted({frame for condition in conditions for frame in condition['frame_indices']})
    fullbody=next((condition for condition in conditions if condition['type']=='fullbody'),None)
    if fullbody is None or len(occupied)>=20:
        raise ValueError('Target feedback requires an existing sparse event plan with fewer than20 time keys')
    def phase(frame):
        return tuple(side for side,frames in hands.items() if frame in frames),frame in feet
    chosen={};intervals=[]
    for side,frames in hands.items():
        failed=sorted(frame for frame in frames if errors[side][frame]>TARGET_HAND_CORRECTION_CM+1e-6)
        runs=[]
        for frame in failed:
            if not runs or frame!=runs[-1][-1]+1:runs.append([])
            runs[-1].append(frame)
        for run in runs:
            peak=max(run,key=lambda frame:errors[side][frame])
            eligible=[frame for frame in occupied if phase(frame)==phase(peak)]
            before=[frame for frame in eligible if frame<=peak];after=[frame for frame in eligible if frame>=peak]
            if not before or not after:
                raise TargetProjectionError('No same-support event keys bracket the target failure',dict(passed=False,peakFrame=peak))
            brackets=sorted({max(before),min(after)})
            excess=max(0.,float(errors[side][peak]-max(errors[side][frame] for frame in brackets)))
            margin=.5+excess
            if margin>=TARGET_HAND_CORRECTION_CM:
                raise TargetProjectionError('Observed interpolation error consumes the target repair budget',dict(passed=False,peakFrame=peak,marginCm=margin))
            interval=dict(side=side,frames=[run[0],run[-1]],peakFrame=peak,
                maximumTargetCorrectionCm=float(errors[side][peak]),bracketFrames=brackets,
                measuredInterpolationOvershootCm=excess,conditioningMarginCm=margin)
            intervals.append(interval)
            for frame in brackets:
                chosen.setdefault(frame,dict(frame=frame,intervals=[]))['intervals'].append(interval)
    if not chosen:
        raise TargetProjectionError('No positional target-budget failure eligible for pose feedback',dict(passed=False,policy=POLICY))
    selected=sorted(chosen.values(),key=lambda row:(-max(item['maximumTargetCorrectionCm'] for item in row['intervals']),row['frame']))[:maximum_frames]
    reports=[]
    for row in sorted(selected,key=lambda item:item['frame']):
        frame=row['frame'];source_contacts=[];target_contacts=[]
        for side,frames in hands.items():
            if frame not in frames:continue
            index=source_names[side.title()+'Hand']
            source_contacts.append(dict(kind='hand',side=side,position=source_points[frame,index],rotation=source_rot[frame,index]))
            target_contacts.append(dict(kind='hand',side=side,position=target_points[side][frame],rotation=target_rot[side][frame]))
        margin=max(item['conditioningMarginCm'] for item in row['intervals'])
        projected,root,receipt=project_target_pose(motion['local_rot_mats'][frame],motion['root_positions'][frame],rig,
            source_contacts,source_reference['root_positions'][frame],skeleton,target_contacts,
            budget_cm=TARGET_HAND_CORRECTION_CM,margin_cm=margin,observer=observer)
        parity=observer.compare_convert(projected,root)
        if not parity['passed']:
            raise TargetProjectionError('Target observer differs from actual production retarget',dict(passed=False,parity=parity))
        rotations=Rotation.from_matrix(projected).as_rotvec().tolist()
        for condition in conditions:
            if frame in condition['frame_indices']:
                index=condition['frame_indices'].index(frame)
                if 'local_joints_rot' in condition:
                    condition['local_joints_rot'][index]=rotations;condition['root_positions'][index]=root.tolist()
                if 'smooth_root_2d' in condition:condition['smooth_root_2d'][index]=root[[0,2]].tolist()
        if frame not in fullbody['frame_indices']:
            fullbody['frame_indices'].append(frame);fullbody['local_joints_rot'].append(rotations);fullbody['root_positions'].append(root.tolist())
        reports.append(dict(**row,projection=receipt,productionRetargetParity=parity,
            originalLocalRotations=Rotation.from_matrix(motion['local_rot_mats'][frame]).as_rotvec().tolist(),
            projectedLocalRotations=rotations))
    order=np.argsort(fullbody['frame_indices']).tolist()
    for key in ('frame_indices','local_joints_rot','root_positions'):fullbody[key]=[fullbody[key][i] for i in order]
    if any(len(condition['frame_indices'])>=20 for condition in conditions):
        raise ValueError('Target feedback exceeds sparse per-type budget')
    receipt=dict(schemaVersion=1,policy=POLICY,reference=identity,semanticName=semantic,
        passed=True,deliverable=False,productionReady=False,scope='candidate-derived generation conditions only',
        targetRepairBudgetCm=TARGET_HAND_CORRECTION_CM,totalKeyUnion=occupied,fullBodyFrames=fullbody['frame_indices'],
        failureIntervals=intervals,selected=reports,constraintsSha256=digest(conditions),
        unchangedSourceAndTargetGates=True,referenceFullBodyPoseCopied=False)
    return conditions,receipt

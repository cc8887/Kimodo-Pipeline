"""Extract one inclusive loop and close its pose/tangent within original budgets.

Only the seam neighbourhood is edited. World travel is removed explicitly and
its exact inverse retained. Tiling is a playback check, not evidence that two
independent generated cycles were stable. All ordinary gait checks still run.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from . import gait, motion


def _angles(a, b):
    return np.rad2deg(np.linalg.norm(Rotation.from_matrix(
        (a @ b.swapaxes(-1, -2)).reshape(-1, 3, 3)).as_rotvec(), axis=-1)).reshape(a.shape[:-2])


def _seam(value, rig):
    local, root = value['local_rot_mats'], value['root_positions']
    fps = float(value['fps'])
    _, points = motion.fk(local, root, rig)
    initial = Rotation.from_matrix(local[1] @ local[0].swapaxes(-1, -2)).as_rotvec()*fps
    final = Rotation.from_matrix(local[-1] @ local[-2].swapaxes(-1, -2)).as_rotvec()*fps
    return dict(poseDegrees=float(_angles(local[-1:], local[:1]).max()),
                bodyMeters=float(np.linalg.norm(points[-1]-points[0], axis=-1).max()),
                velocityMps=float(np.linalg.norm((points[1]-points[0]-points[-1]+points[-2])*fps,axis=-1).max()),
                angularVelocityRps=float(np.linalg.norm(initial-final, axis=-1).max()))


def tile_loop(loop, rig, *, cycles=4, displacement):
    """Tile unique keys; include the terminal key once and restore travel."""
    if not isinstance(cycles, int) or isinstance(cycles, bool) or cycles < 3:
        raise ValueError('Playback verification requires at least three repeats')
    motion.validate_motion(loop, rig)
    period=len(loop['root_positions'])-1
    if period<3:raise ValueError('A loop needs at least three intervals')
    if (not np.allclose(loop['local_rot_mats'][0],loop['local_rot_mats'][-1],atol=1e-10,rtol=0)
            or not np.allclose(loop['root_positions'][0],loop['root_positions'][-1],atol=1e-10,rtol=0)):
        raise ValueError('Cannot tile an open loop by silently replacing its endpoint')
    displacement=np.asarray(displacement,dtype=float)
    if displacement.shape!=(3,) or not np.isfinite(displacement).all():
        raise ValueError('Invalid loop travel displacement')
    phase=np.arange(cycles*period+1)
    local=loop['local_rot_mats'][phase%period].copy()
    root=loop['root_positions'][phase%period]+phase[:,None]/period*displacement
    return motion.complete(local,root,rig,float(loop['fps']))


def crop_loop(source, rig, start, end_inclusive, *, seam_frames=3, close=True,match_tangents=True,
              include_correction_paths=False):
    """Return a diagnostic single loop plus source-bound extraction evidence.

    This operation never certifies support or target skin. A geometrically
    closed loop must still pass authoring, target, file and runtime checks.
    """
    motion.validate_motion(source,rig)
    if any(not isinstance(value,bool) for value in (close,match_tangents,include_correction_paths)):
        raise ValueError('Loop closure and reporting options must be boolean')
    if (any(not isinstance(v,int) or isinstance(v,bool) for v in (start,end_inclusive,seam_frames))
            or not 0<=start<end_inclusive<len(source['root_positions'])
            or seam_frames<2 or seam_frames*2>=end_inclusive-start):
        raise ValueError('Invalid inclusive loop span or seam neighbourhood')
    local=source['local_rot_mats'][start:end_inclusive+1].copy()
    original_root=source['root_positions'][start:end_inclusive+1].copy()
    count=len(local);fps=float(source['fps'])
    displacement=np.zeros(3);displacement[[0,2]]=original_root[-1,[0,2]]-original_root[0,[0,2]]
    removed=np.zeros_like(original_root)
    removed[:,[0,2]]=original_root[0,[0,2]]+np.linspace(0,1,count)[:,None]*displacement[[0,2]]
    root=original_root-removed
    before=motion.complete(local.copy(),root.copy(),rig,fps)
    if close:
        # Shortest geodesic endpoint mean and equal angular increments at both
        # sides. Decay each endpoint correction smoothly into the untouched
        # interior; no Euler-angle interpolation or duplicated zero-length key.
        delta=Rotation.from_matrix(local[-1] @ local[0].swapaxes(-1,-2)).as_rotvec()
        shared=Rotation.from_rotvec(.5*delta).as_matrix() @ local[0]
        first=Rotation.from_matrix(local[1] @ shared.swapaxes(-1,-2)).as_rotvec()
        last=Rotation.from_matrix(shared @ local[-2].swapaxes(-1,-2)).as_rotvec()
        tangent=.5*(first+last)
        target={0:shared,1:Rotation.from_rotvec(tangent).as_matrix()@shared,
                count-2:Rotation.from_rotvec(-tangent).as_matrix()@shared,count-1:shared}
        shared_root=.5*(root[0]+root[-1])
        root_step=.5*(root[1]-shared_root+shared_root-root[-2])
        root_target={0:shared_root,1:shared_root+root_step,count-2:shared_root-root_step,count-1:shared_root}
        original_local=local.copy();original_residual=root.copy()
        if not match_tangents:
            target={0:shared,count-1:shared}
            root_target={0:shared_root,count-1:shared_root}
        for key in ((1,count-2) if match_tangents else (0,count-1)):
            correction=Rotation.from_matrix(target[key]@original_local[key].swapaxes(-1,-2)).as_rotvec()
            translation=root_target[key]-original_residual[key]
            direction=1 if key<=1 else -1
            for distance in range(seam_frames):
                frame=key+direction*distance
                x=distance/seam_frames
                weight=1-(x*x*x*(10+x*(-15+6*x)))
                local[frame]=Rotation.from_rotvec(weight*correction).as_matrix()@original_local[frame]
                root[frame]=original_residual[frame]+weight*translation
        for key in target:local[key]=target[key];root[key]=root_target[key]
    loop=motion.complete(local,root,rig,fps)
    angles=_angles(local,before['local_rot_mats'])
    root_change=np.linalg.norm(root-before['root_positions'],axis=-1)
    original_feet,_=gait.foot_landmarks(before,rig);new_feet,_=gait.foot_landmarks(loop,rig)
    foot_change=np.linalg.norm(new_feet-original_feet,axis=-1)
    feet=[rig['jointNames'].index(side+'Foot') for side in ('Left','Right')]
    original_joints=motion.fk(before['local_rot_mats'],before['root_positions'],rig)[1]
    new_joints=motion.fk(local,root,rig)[1]
    ankle_change=np.linalg.norm(new_joints[:,feet]-original_joints[:,feet],axis=-1)
    budget=dict(maximumRotationDegrees=float(angles.max()),maximumRootMeters=float(root_change.max()),
                maximumFootMeters=float(foot_change.max()))
    if (budget['maximumRotationDegrees']>20. or budget['maximumRootMeters']>.05
            or budget['maximumFootMeters']>.15):
        raise ValueError('Loop seam closure exceeds original correction budgets: '+str(budget))
    report=dict(schemaVersion=1,policy='single-loop-local-geodesic-closure-v1',productionReady=False,
        sourceMotionSha256=gait.audit_candidate(source,rig)['sourceMotionSha256'],
        sourceSpan=[start,end_inclusive],frameCount=count,uniqueKeyCount=count-1,
        durationSeconds=(count-1)/fps,seamFrames=seam_frames,rotationAndRootTangentsMatched=bool(close and match_tangents),
        removedPlanarTrajectory=dict(positions=removed.tolist(),displacement=displacement.tolist()),
        correctionBudget=budget,before=_seam(before,rig),after=_seam(loop,rig),
        stableGeneratedCyclesProven=False,
        requires=['independent_authoring_gate','target_skin_gate','final_fbx_readback','godot_runtime'])
    if include_correction_paths:
        report['correctionPaths']=dict(rotationDegrees=angles.tolist(),ankleMeters=ankle_change.tolist(),rootMeters=root_change.tolist())
    return loop,report

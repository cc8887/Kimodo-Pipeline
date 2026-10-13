"""Bounded, endpoint-anchored final loop repair, before independent acceptance.

Source seams are diagnostic inputs. Acceptance measures the repaired keys,
including both SLERP and the consumer's NLERP, with unchanged gait limits.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from . import gait, motion
from .loop_segment import _angles, _seam


def interpolation_seam(value, rig, method='nlerp'):
    """Independent left/right subframe FK limits; no circular stencil identity."""
    if method not in ('slerp', 'nlerp'):
        raise ValueError('Unknown playback interpolation')
    local, root = value['local_rot_mats'], value['root_positions']
    epsilon = 1.e-4
    def sample(a, b, t):
        if method == 'slerp':
            step = Rotation.from_matrix(local[b] @ local[a].swapaxes(-1, -2)).as_rotvec()
            rotations = Rotation.from_rotvec(t*step).as_matrix() @ local[a]
        else:
            qa, qb = Rotation.from_matrix(local[a]).as_quat(), Rotation.from_matrix(local[b]).as_quat()
            qb = qb*np.where(np.sum(qa*qb, axis=-1, keepdims=True) < 0, -1., 1.)
            q = (1-t)*qa+t*qb
            rotations = Rotation.from_quat(q/np.linalg.norm(q, axis=-1, keepdims=True)).as_matrix()
        return motion.fk(rotations[None], ((1-t)*root[a]+t*root[b])[None], rig)[1][0]
    first = (sample(0, 1, epsilon)-sample(0, 1, 0))/epsilon*float(value['fps'])
    last = (sample(-2, -1, 1)-sample(-2, -1, 1-epsilon))/epsilon*float(value['fps'])
    differences = np.linalg.norm(first-last, axis=-1)
    worst = int(np.argmax(differences))
    return dict(velocityMps=float(differences[worst]), bone=rig['jointNames'][worst],
                method=method, subframeFraction=epsilon)


def _paths(before, after, rig):
    rotation = _angles(after['local_rot_mats'], before['local_rot_mats'])
    a = motion.fk(before['local_rot_mats'], before['root_positions'], rig)[1]
    b = motion.fk(after['local_rot_mats'], after['root_positions'], rig)[1]
    feet = [rig['jointNames'].index(side+'Foot') for side in ('Left', 'Right')]
    return dict(rotationDegrees=rotation, ankleMeters=np.linalg.norm(b[:, feet]-a[:, feet], axis=-1),
                rootMeters=np.linalg.norm(after['root_positions']-before['root_positions'], axis=-1))


def repair_loop_seam(source, rig, *, window_frames=3, anchor='last', prior_paths=None,
                     playback_displacement=None, expected_speed_mps=None):
    """Fix one endpoint and ease both sides towards common bounded tangents.

    Prefer the least damped tangent that satisfies all seam metrics and budgets.
    Damping is a fallback for finite-key FK curvature, not a changed threshold.
    The middle keys and the chosen endpoint remain unchanged. Contact/skin/style
    acceptance belongs to the subsequent independent gates.
    """
    motion.validate_motion(source, rig)
    count = len(source['root_positions'])
    maximum_window = (count-3)//2  # Leave at least one untouched middle key.
    if (not isinstance(window_frames, int) or isinstance(window_frames, bool)
            or window_frames < 2 or window_frames > maximum_window):
        raise ValueError('Invalid or overlapping seam repair windows')
    if anchor not in ('first', 'last'):
        raise ValueError('Seam anchor must be first or last')
    if playback_displacement is not None:
        playback_displacement = np.asarray(playback_displacement, dtype=float)
        if playback_displacement.shape != (3,) or not np.isfinite(playback_displacement).all():
            raise ValueError('Invalid authoring travel for seam acceptance')
    def audit_playback(value):
        from .loop_segment import tile_loop
        return gait.audit_candidate(tile_loop(value, rig, displacement=playback_displacement), rig, expected_speed_mps)
    limits = dict(rotationDegrees=20., ankleMeters=.15, rootMeters=.05)
    zero = _paths(source, source, rig)
    prior = zero if prior_paths is None else {k: np.asarray(prior_paths[k], dtype=float) for k in zero}
    if any(prior[k].shape != zero[k].shape or not np.isfinite(prior[k]).all()
           or np.any(prior[k] < 0) for k in zero):
        raise ValueError('Invalid same-frame prior correction paths')
    policy = gait._policy(None)
    def assess(value):
        seam = _seam(value, rig)
        playback = {method: interpolation_seam(value, rig, method) for method in ('slerp', 'nlerp')}
        passed = (seam['poseDegrees'] <= policy['maximumSeamRotationDegrees']
                  and seam['bodyMeters'] <= policy['maximumSeamBodyMeters']
                  and abs(value['root_positions'][-1, 1]-value['root_positions'][0, 1])
                      <= policy['maximumSeamRootHeightMeters']
                  and seam['velocityMps'] <= policy['maximumSeamVelocityMps']
                  and seam['angularVelocityRps'] <= policy['maximumSeamAngularVelocityRps']
                  and all(p['velocityMps'] <= policy['maximumSeamVelocityMps'] for p in playback.values()))
        # Final loops need exact closure as well as quality thresholds, for tiling.
        passed = passed and np.allclose(value['local_rot_mats'][0], value['local_rot_mats'][-1], atol=1e-10, rtol=0)
        passed = passed and np.allclose(value['root_positions'][0], value['root_positions'][-1], atol=1e-10, rtol=0)
        return dict(passSeam=bool(passed), discrete=seam, interpolation=playback)
    before = assess(source)
    unchanged_audit = audit_playback(source) if before['passSeam'] and playback_displacement is not None else None
    if (before['passSeam'] and all(prior[k].max() <= limits[k] for k in limits)
            and (unchanged_audit is None or unchanged_audit['pass'])):
        return motion.complete(source['local_rot_mats'].copy(), source['root_positions'].copy(), rig, float(source['fps'])), dict(
            policy='anchored-eased-loop-seam-v1', applied=False, passed=True, before=before, after=before,
            correctionPaths={k: v.tolist() for k, v in zero.items()},
            cumulativeCorrectionPaths={k: v.tolist() for k, v in prior.items()}, attempts=[], productionReady=False)
    local, root = source['local_rot_mats'], source['root_positions']
    key = 0 if anchor == 'first' else count-1
    shared, shared_root = local[key].copy(), root[key].copy()
    first = Rotation.from_matrix(local[1] @ shared.swapaxes(-1, -2)).as_rotvec()
    last = Rotation.from_matrix(shared @ local[-2].swapaxes(-1, -2)).as_rotvec()
    tangent = .5*(first+last)
    root_step = .5*(root[1]-root[-2])
    attempts = []
    fallback = None
    for scale in (1., .85, .7, .55, .4, .25, 0.):
        for width in range(window_frames, maximum_window+1):
            out_local, out_root = local.copy(), root.copy()
            targets = {0: shared, 1: Rotation.from_rotvec(scale*tangent).as_matrix() @ shared,
                       count-2: Rotation.from_rotvec(-scale*tangent).as_matrix() @ shared, count-1: shared}
            roots = {0: shared_root, 1: shared_root+scale*root_step,
                     count-2: shared_root-scale*root_step, count-1: shared_root}
            for adjacent, direction in ((1, 1), (count-2, -1)):
                delta = Rotation.from_matrix(targets[adjacent] @ local[adjacent].swapaxes(-1, -2)).as_rotvec()
                translation = roots[adjacent]-root[adjacent]
                for distance in range(width):
                    frame = adjacent+direction*distance
                    x = distance/width
                    weight = 1-x*x*x*(10+x*(-15+6*x))
                    out_local[frame] = Rotation.from_rotvec(weight*delta).as_matrix() @ local[frame]
                    out_root[frame] = root[frame]+weight*translation
            for frame in targets:
                out_local[frame], out_root[frame] = targets[frame], roots[frame]
            value = motion.complete(out_local, out_root, rig, float(source['fps']))
            paths = _paths(source, value, rig)
            cumulative = {k: prior[k]+paths[k] for k in paths}
            old_feet = gait.foot_landmarks(source, rig)[0]
            new_feet = gait.foot_landmarks(value, rig)[0]
            foot_peak = float(np.linalg.norm(new_feet-old_feet, axis=-1).max())
            assessment = assess(value)
            peaks = {k: float(v.max()) for k, v in cumulative.items()}
            allowed = all(peaks[k] <= limits[k] for k in limits) and foot_peak <= .15
            attempts.append(dict(windowFrames=width, tangentScale=scale, budgetPass=allowed,
                                 cumulativePeaks=peaks, maximumFootMeters=foot_peak, **assessment))
            if assessment['passSeam'] and allowed:
                audit = audit_playback(value) if playback_displacement is not None else None
                if audit is not None:
                    attempts[-1]['playbackGate'] = dict(passed=audit['pass'],
                        failedChecks=[c for c in audit['checks'] if not c['pass']],
                        contactFrames={side: audit['detection']['measurements'][side]['contactFrames']
                                       for side in ('left', 'right')})
                report = dict(policy='anchored-eased-loop-seam-v1', applied=True, passed=True,
                    anchor=anchor, anchorFrame=key, windowFrames=width, tangentScale=scale,
                    easing='quintic smootherstep correction decay', before=before, after=assessment,
                    correctionPaths={k: v.tolist() for k, v in paths.items()},
                    cumulativeCorrectionPaths={k: v.tolist() for k, v in cumulative.items()},
                    cumulativePeaks=peaks, limits=limits, attempts=attempts, productionReady=False,
                    requires=['independent_contact_and_style_gate', 'target_gate', 'final_fbx', 'runtime'])
                if audit is None or audit['pass']:
                    report['playbackGatePass'] = None if audit is None else True
                    return value, report
                if fallback is None:
                    fallback = value, report
    if fallback is not None:
        # Retain a seam-fixed diagnostic, never approve a contact failure.
        value, report = fallback
        report['playbackGatePass'] = False
        report['selection'] = 'first bounded seam pass; all contact-aware alternatives rejected'
        return value, report
    error = ValueError('Anchored loop seam repair has no candidate within original seam and correction limits')
    error.seam_report = dict(policy='anchored-eased-loop-seam-v1', passed=False, before=before,
                             anchor=anchor, limits=limits, attempts=attempts, productionReady=False)
    raise error

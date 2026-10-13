"""Joint loop/contact cleanup; constraints propose, independent gates accept."""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation

from . import gait, motion
from .gait_constraints import solve_foot_ik
from .loop_segment import tile_loop, _seam
from .loop_seam import repair_loop_seam, interpolation_seam, _paths


def solve_loop_contacts(source, rig, displacement, support_mask, *, prior_paths=None, max_evaluations=100, seam_weight=1., support_weight=1., near_ground_weight=1., optimize_shared_anchor=False, optimize_pelvis=False):
    """Keep the upper-body repair and jointly solve fixed-length legs and Root.

    The support mask must come from a verified phase plan, never a synthesized
    approval curve. Output is always remeasured from FK. The canonical seam is
    a measured interior left touchdown, falling back to the verified plan.
    Its foot anchor is restored before it is fixed.
    """
    motion.validate_motion(source, rig)
    period = len(source['root_positions'])-1
    if period < 8:
        raise ValueError('Joint loop repair requires at least eight intervals')
    displacement = np.asarray(displacement, dtype=float)
    mask = np.asarray(support_mask)
    if displacement.shape != (3,) or not np.isfinite(displacement).all() or abs(displacement[1]) > 1e-10:
        raise ValueError('Joint loop repair needs explicit planar travel')
    if mask.shape != (period, 2, 2) or not np.isin(mask, (0, 1)).all() or not mask.any(axis=(0, 2)).all():
        raise ValueError('Joint loop repair requires independent bilateral support intent')
    if type(max_evaluations) is not int or not 1 <= max_evaluations <= 500:
        raise ValueError('Invalid bounded optimizer iteration count')
    if not np.isfinite(seam_weight) or not 1. <= seam_weight <= 10.:
        raise ValueError('Invalid bounded seam optimization weight')
    if not np.isfinite(support_weight) or not 1. <= support_weight <= 10.:
        raise ValueError('Invalid bounded support optimization weight')
    if not np.isfinite(near_ground_weight) or not 1. <= near_ground_weight <= 10.:
        raise ValueError('Invalid bounded near-ground optimization weight')
    if type(optimize_shared_anchor) is not bool or type(optimize_pelvis) is not bool:
        raise ValueError('Shared endpoint optimization must be boolean')
    if (not np.allclose(source['local_rot_mats'][0], source['local_rot_mats'][-1], atol=1e-10, rtol=0)
            or not np.allclose(source['root_positions'][0], source['root_positions'][-1], atol=1e-10, rtol=0)):
        raise ValueError('Joint phase rotation requires a closed input, not endpoint replacement')
    mask = mask.astype(bool)
    left = mask[:, 0].any(axis=-1)
    starts = np.flatnonzero(left & ~np.roll(left, 1))
    if len(starts) != 1:
        raise ValueError('Joint repair requires one unambiguous left support phase')
    observed = gait.audit_candidate(tile_loop(source, rig, displacement=displacement), rig)
    touchdowns = [e['frame'] % period for e in observed['detection']['events']
                  if e['kind'] == 'touchdown' and e['side'] == 'left' and e['frame'] >= period]
    phase = int(touchdowns[0]) if touchdowns else int(starts[0])
    indices = (np.arange(period+1)+phase) % period
    original = motion.complete(source['local_rot_mats'][indices].copy(), source['root_positions'][indices].copy(),
                               rig, float(source['fps']))
    mask = np.roll(mask, -phase, axis=0)
    stable_mask = mask & np.roll(mask, 1, axis=0) & np.roll(mask, -1, axis=0)
    zero = _paths(original, original, rig)
    prior = zero
    if prior_paths is not None:
        try:
            raw_prior = {k: np.asarray(prior_paths[k], dtype=float) for k in zero}
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError('Invalid joint repair correction history') from error
        if any(raw_prior[k].shape != zero[k].shape for k in zero):
            raise ValueError('Invalid joint repair correction history')
        prior = {k: v[indices] for k, v in raw_prior.items()}
    if any(prior[k].shape != zero[k].shape or not np.isfinite(prior[k]).all() or np.any(prior[k] < 0) for k in zero):
        raise ValueError('Invalid joint repair correction history')
    limits = dict(rotationDegrees=20., ankleMeters=.15, rootMeters=.05)
    if any(prior[k].max() >= limits[k] for k in limits):
        raise ValueError('Joint repair exhausted original correction budgets')
    unchanged_seam = _seam(original, rig)
    unchanged_interpolation = {m: interpolation_seam(original, rig, m) for m in ('slerp', 'nlerp')}
    if (observed['pass'] and unchanged_seam['velocityMps'] <= .75 and unchanged_seam['angularVelocityRps'] <= 6.
            and all(v['velocityMps'] <= .75 for v in unchanged_interpolation.values())):
        return original, dict(policy='joint-loop-contact-seam-v1', passed=True, applied=False, productionReady=False,
            canonicalPhase=phase, sourceFrameIndices=indices.tolist(), supportIntent=mask.tolist(),
            cumulativePeaks={k: float(v.max()) for k, v in prior.items()}, limits=limits,
            cumulativeCorrectionPaths={k: v.tolist() for k, v in prior.items()},
            seam=unchanged_seam, interpolation=unchanged_interpolation, after=observed,
            sourceMotionSha256=gait._array_hash(source['local_rot_mats'], source['root_positions'], source['fps']))
    names = {n: i for i, n in enumerate(rig['jointNames'])}
    bones = [names[side+suffix] for side in ('Left', 'Right') for suffix in ('Leg', 'Shin', 'Foot')]
    bones += [names[side+'ToeBase'] for side in ('Left', 'Right') if side+'ToeBase' in names]
    lower = set(bones)
    toes = []
    for side in ('Left', 'Right'):
        foot = names[side+'Foot']
        chain = {}
        for i in range(len(names)):
            ancestor = i
            while ancestor >= 0 and ancestor != foot:
                ancestor = rig['parents'][ancestor]
            if ancestor == foot:
                lower.add(i)
                if i != foot:
                    chain[i] = original['local_rot_mats'][0, i]
        toes.append(chain)
    upper = [i for i in range(len(names)) if i not in lower]
    proposed, upper_report = repair_loop_seam(original, rig)
    local = original['local_rot_mats'].copy()
    local[:, upper] = proposed['local_rot_mats'][:, upper]
    root = original['root_positions'].copy()
    upper_motion = motion.complete(local.copy(), root.copy(), rig, float(source['fps']))
    cumulative = {k: prior[k]+_paths(original, upper_motion, rig)[k] for k in prior}
    original_global, original_joints = motion.fk(original['local_rot_mats'], original['root_positions'], rig)
    # Match the physical foot anchors before fixing the final seam pose. Hips
    # tangent cleanup must not move the stance feet through the floor.
    for side, label in enumerate(('left', 'right')):
        foot = names[label.title()+'Foot']
        local[0] = solve_foot_ik(local[0], root[0], rig, label, original_joints[0, foot],
                                  original_global[0, foot], toes[side])
    local[-1] = local[0]
    anchored = motion.complete(local.copy(), root.copy(), rig, float(source['fps']))
    anchor_joints = motion.fk(local, root, rig)[1]
    upper_children = []
    anchored_global = motion.fk(local, root, rig)[0]
    if optimize_pelvis:
        pelvis = names['Hips']
        bones.append(pelvis)
        upper_children = [i for i, parent in enumerate(rig['parents']) if parent == pelvis and i not in lower]
    ankle_ids = [names[side+'Foot'] for side in ('Left', 'Right')]
    anchor_paths = _paths(upper_motion, anchored, rig)
    cumulative = {k: cumulative[k]+anchor_paths[k] for k in prior}
    if any(cumulative[k].max() >= limits[k] for k in limits):
        raise ValueError('Anchored joint repair exhausted original correction budgets')
    feet, geometry = gait.foot_landmarks(original, rig)
    definitions = geometry['definitions']
    fps = float(source['fps'])
    travel = np.arange(period+1)[:, None]/period*displacement
    targets = feet+travel[:, None, None, :]
    near = feet[:-1, :, :, 1] <= gait.DEFAULT_POLICY['contactLeaveHeightMeters']
    stride = len(bones)*3+3
    first_variable = 0 if optimize_shared_anchor else 1
    frames = np.arange(first_variable, period)
    variable_count = len(frames)*stride
    columns = [np.array([], dtype=int) for _ in range(period+1)]
    for index, frame in enumerate(frames):
        columns[frame] = np.arange(index*stride, (index+1)*stride)
    columns[period] = columns[0]
    sparsity = None
    ankle_penalty, ankle_margin = 10000., 1e-5
    def decode(x):
        values = x.reshape(len(frames), stride)
        rotations, roots = local.copy(), root.copy()
        increments = values[:, :len(bones)*3].reshape(len(frames), len(bones), 3)
        available = np.maximum(0., np.deg2rad(20.-cumulative['rotationDegrees'][frames[:, None], bones]-1e-4))
        sizes = np.linalg.norm(increments, axis=-1)
        increments *= np.minimum(1., available/np.maximum(sizes, 1e-12))[:, :, None]
        for i, bone in enumerate(bones):
            rotations[frames, bone] = Rotation.from_rotvec(increments[:, i]).as_matrix() @ local[frames, bone]
        if optimize_pelvis:
            global_pelvis = motion.fk(rotations, roots, rig)[0][:, pelvis]
            for child in upper_children:
                rotations[:, child] = global_pelvis.swapaxes(-1, -2) @ anchored_global[:, child]
        root_increments = values[:, -3:].copy()
        available_root = np.maximum(0., .05-cumulative['rootMeters'][frames]-1e-8)
        root_sizes = np.linalg.norm(root_increments, axis=-1)
        root_increments *= np.minimum(1., available_root/np.maximum(root_sizes, 1e-12))[:, None]
        roots[frames] += root_increments
        rotations[-1], roots[-1] = rotations[0], roots[0]
        return rotations, roots
    def positions(rotations, roots):
        global_rot, joints = motion.fk(rotations, roots, rig)
        points = np.empty((period+1, 2, 2, 3))
        for side, label in enumerate(('left', 'right')):
            for sample, key in enumerate(('heel', 'toe')):
                d = definitions[label][key]
                joint = names[d['joint']]
                points[:, side, sample] = joints[:, joint]+global_rot[:, joint] @ np.asarray(d['offset'])
        return joints, points
    def residual(x, build=False):
        nonlocal sparsity
        rotations, roots = decode(x)
        joints, points = positions(rotations, roots)
        world = points+travel[:, None, None, :]
        unique = world[:-1]
        velocity = (np.roll(unique, -1, axis=0)-np.roll(unique, 1, axis=0))*fps/2
        velocity[0] += displacement*fps/2
        velocity[-1] += displacement*fps/2
        chunks, dependencies = [], []
        def add(value, frames):
            chunks.append(np.asarray(value).reshape(-1))
            dependencies.append(np.unique(np.concatenate([columns[f] for f in frames])))
        for frame in frames:
            values = x[columns[frame]]
            add(values[:-3]*.06, [frame])
            add(values[-3:]/.05*.06, [frame])
            # Norm barriers preserve the same-frame, same-bone budget.
            remaining = 20.-cumulative['rotationDegrees'][frame, bones]
            add(np.maximum(0., np.rad2deg(np.linalg.norm(values[:-3].reshape(-1, 3), axis=-1))-remaining+1e-4)*100, [frame])
            ankle_change = np.linalg.norm(joints[frame, ankle_ids]-anchor_joints[frame, ankle_ids], axis=-1)
            add(np.maximum(0., ankle_change+cumulative['ankleMeters'][frame]-.15+ankle_margin)*ankle_penalty, [frame])
        for frame in range(period+1):
            actual = 0 if frame == period else frame
            add((world[frame]-targets[frame])[mask[actual]]/.001, [frame])
            add(np.minimum(points[frame, :, :, 1]-.001, 0)/.003, [frame])
            add((points[frame]-feet[frame])/.05*.1, [frame])
            if frame < period:
                neighbours = [(frame-1) % period, (frame+1) % period]
                add(velocity[frame][stable_mask[frame]]/.03*support_weight, neighbours)
                add(velocity[frame][near[frame]]/.25*.2*support_weight*near_ground_weight, neighbours)
        add((joints[1]-joints[0]-joints[-1]+joints[-2])*fps/.3*5*seam_weight, [0, 1, period-1])
        initial = Rotation.from_matrix(rotations[1] @ rotations[0].swapaxes(-1, -2)).as_rotvec()
        final = Rotation.from_matrix(rotations[-1] @ rotations[-2].swapaxes(-1, -2)).as_rotvec()
        add((initial-final)*fps/3*3*seam_weight, [0, 1, period-1])
        epsilon = 1.e-4
        left_local = Rotation.from_rotvec((1-epsilon)*final).as_matrix() @ rotations[-2]
        right_local = Rotation.from_rotvec(epsilon*initial).as_matrix() @ rotations[0]
        left_root = (1-epsilon)*roots[-1]+epsilon*roots[-2]
        right_root = (1-epsilon)*roots[0]+epsilon*roots[1]
        left_points = motion.fk(left_local[None], left_root[None], rig)[1][0]
        right_points = motion.fk(right_local[None], right_root[None], rig)[1][0]
        add((right_points-joints[0]-joints[-1]+left_points)*fps/epsilon/.3*5*seam_weight, [0, 1, period-1])
        if build:
            sparsity = lil_matrix((sum(len(c) for c in chunks), variable_count), dtype=int)
            offset = 0
            for chunk, cols in zip(chunks, dependencies):
                sparsity[offset:offset+len(chunk), cols] = 1
                offset += len(chunk)
            sparsity = sparsity.tocsr()
        return np.concatenate(chunks)
    bound = np.empty((len(frames), stride))
    for index, frame in enumerate(frames):
        bound[index, :-3] = np.repeat(np.deg2rad(20.-cumulative['rotationDegrees'][frame, bones]), 3)
        bound[index, -3:] = .05-cumulative['rootMeters'][frame]
    residual(np.zeros(variable_count), build=True)
    solution = least_squares(residual, np.zeros(variable_count), bounds=(-bound.ravel(), bound.ravel()),
                             jac_sparsity=sparsity, max_nfev=max_evaluations, ftol=1e-7, xtol=1e-7, gtol=1e-7)
    main_evaluations = solution.nfev
    rotations, roots = decode(solution.x)
    first_paths = _paths(anchored, motion.complete(rotations, roots, rig, fps), rig)
    polish_evaluations = 0
    if np.max(cumulative['ankleMeters']+first_paths['ankleMeters']) > .15:
        # Soft FK barriers can miss a strict budget by micrometers. Re-solve
        # with a reserved margin; never round, tolerate or waive the limit.
        ankle_penalty, ankle_margin = 1e6, 1e-4
        solution = least_squares(residual, solution.x, bounds=(-bound.ravel(), bound.ravel()),
                                 jac_sparsity=sparsity, max_nfev=min(max_evaluations, 100),
                                 ftol=1e-7, xtol=1e-7, gtol=1e-7)
        polish_evaluations = solution.nfev
        rotations, roots = decode(solution.x)
    result = motion.complete(rotations, roots, rig, fps)
    paths = _paths(anchored, result, rig)
    cumulative = {k: cumulative[k]+paths[k] for k in cumulative}
    peaks = {k: float(v.max()) for k, v in cumulative.items()}
    audit = gait.audit_candidate(tile_loop(result, rig, displacement=displacement), rig,
                                 float(np.linalg.norm(displacement))/ (period/fps))
    interpolated = {method: interpolation_seam(result, rig, method) for method in ('slerp', 'nlerp')}
    seam = _seam(result, rig)
    passed = (audit['pass'] and all(peaks[k] <= limits[k] for k in limits)
              and seam['velocityMps'] <= .75 and seam['angularVelocityRps'] <= 6.
              and all(v['velocityMps'] <= .75 for v in interpolated.values()))
    report = dict(policy='joint-loop-contact-seam-v1', passed=bool(passed), applied=True, productionReady=False,
        sourceMotionSha256=gait._array_hash(source['local_rot_mats'], source['root_positions'], source['fps']),
        canonicalPhase=phase, sourceFrameIndices=indices.tolist(), supportIntent=mask.tolist(),
        upperSeamProposal=upper_report, evaluations=main_evaluations+polish_evaluations, cost=float(solution.cost), seamOptimizationWeight=float(seam_weight), supportOptimizationWeight=float(support_weight), nearGroundOptimizationWeight=float(near_ground_weight),
        optimizerConverged=bool(solution.success), seam=seam, interpolation=interpolated,
        budgetPolishEvaluations=polish_evaluations,
        cumulativePeaks=peaks, limits=limits, cumulativeCorrectionPaths={k: v.tolist() for k, v in cumulative.items()},
        after=audit, preservesBoneLengths=True, upperBodyTracksPreservedAfterRepair=not optimize_pelvis,
        fixedAnchorAfterFootProjection=not optimize_shared_anchor,
        sharedEndpointPoseOptimized=optimize_shared_anchor, pelvisOptimized=optimize_pelvis,
        upperBodyGlobalOrientationsPreserved=optimize_pelvis, endpointEqualityHardConstraint=True)
    return result, report

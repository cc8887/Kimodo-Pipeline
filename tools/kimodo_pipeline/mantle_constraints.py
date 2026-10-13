"""Sparse semantic contact conditions; style remains in unconstrained joints.

The ALS adapter supplies its current event times and reachable reference rig.
This does not certify a new timing contract or relax author/target acceptance.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
from scipy.spatial.transform import Rotation

SPARSE_POLICY = 'sparse-contact-v1'


def contact_plan(spec, duration, count):
    if not np.isfinite(duration) or duration <= 0 or count < 3:
        raise ValueError('Invalid contact-plan time domain')
    contacts = {}
    for side in spec['hands']:
        if side not in ('left', 'right') or side in contacts:
            raise ValueError('Invalid or repeated support hand')
        begin, end = spec.get('handContactSeconds', {}).get(side, spec['contactSeconds'])
        if not 0 <= begin < end < duration:
            raise ValueError('Contact event is outside the delivery timeline')
        contacts[side] = dict(beginSeconds=float(begin), endSeconds=float(end),
            keyFrames=sorted(set(round(t / duration * (count-1)) for t in np.linspace(begin, end, 5))))
    ground_begin, ground_end = spec['groundedSeconds']
    if not 0 <= ground_begin <= ground_end <= duration or not contacts:
        raise ValueError('Invalid landing event or missing support')
    feet = sorted(set(round(t / duration * (count-1)) for t in (ground_end, (ground_end+duration)/2, duration)))
    # All effectors share this sparse event set. Unlike per-joint dense masks,
    # their union also remains below the model's recommended 20-key budget.
    hips = sorted({0, count-1, round(ground_begin/duration*(count-1)),
                   *feet, *(f for item in contacts.values() for f in item['keyFrames'])})
    if len(hips) >= 20:
        raise ValueError('Contact event union exceeds sparse conditioning budget')
    return dict(schemaVersion=1, policy=SPARSE_POLICY, durationSeconds=float(duration), frames=count,
        contacts=contacts, fullBodyFrames=[0, count-1], hipFrames=hips, landingFootFrames=feet,
        timingScope='existing ALS compatibility events; not inferred from prompt style',
        guarantee='conditioning only; independently validate every intermediate frame')


def materialize_sparse_constraints(params, reference, rig):
    from .mantle import POLICY, profile, reference_input
    from .motion import resample
    name = params['semantic']
    native, _, identity = reference_input(name)
    count = int(params['seconds'] * params['fps'])
    plan = contact_plan(profile(name), native['playLength'], count)
    guide = resample(reference, count, rig)
    rotations = Rotation.from_matrix(guide['local_rot_mats'].reshape(-1,3,3)).as_rotvec().reshape(count,-1,3)
    def condition(kind, frames):
        index = np.asarray(frames, dtype=int)
        return dict(type=kind, frame_indices=list(frames), local_joints_rot=rotations[index].tolist(),
                    root_positions=guide['root_positions'][index].tolist())
    conditions = [condition('fullbody', plan['fullBodyFrames'])]
    hips = condition('end-effector', plan['hipFrames'])
    hips['joint_names'] = ['Hips']
    conditions.append(hips)
    for side, contact in plan['contacts'].items():
        conditions.append(condition(side+'-hand', contact['keyFrames']))
    for side in ('left', 'right'):
        conditions.append(condition(side+'-foot', plan['landingFootFrames']))
    fingerprint = hashlib.sha256(json.dumps(plan, sort_keys=True, separators=(',',':')).encode()).hexdigest()
    receipt = dict(schemaVersion=1, policy=POLICY, conditioningPolicy=SPARSE_POLICY, semanticName=name,
        reference=identity, contactPlan=plan, contactPlanSha256=fingerprint,
        generationFrames=count, deliveryFrames=native['sampledKeyCount'], handPositionAndOrientation=True,
        fullBodyFrames=plan['fullBodyFrames'], generationHipTrajectoryFrames=plan['hipFrames'],
        generationContactFrames={side:item['keyFrames'] for side,item in plan['contacts'].items()},
        generationFinishSupportFrames=plan['landingFootFrames'],
        implicitEffectorRootConstraints=['smoothed root XZ', 'root Y', 'global heading'],
        rootConstraintSource='reachable reference at semantic event keys; not an unconstrained body trajectory',
        hipAccuracy='soft inference constraint; current native cleanup does not guarantee generic Hips pins',
        unchangedAcceptance=['source contact and pelvis', 'target repair budget', 'final FBX', 'runtime skin and pose'],
        productionReady=False)
    return conditions, receipt

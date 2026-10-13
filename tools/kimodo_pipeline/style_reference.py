"""Optional generated-style upper pose for sparse boundary constraints only."""
from __future__ import annotations

import hashlib
import json
import math

import numpy as np
from scipy.spatial.transform import Rotation

from .gait import _array_hash, foot_landmarks
from .motion import complete, fk, validate_motion


POLICY = 'generated-upper-style-constraint-reference-v1'


def _rig_binding(rig):
    names, parents = rig.get('jointNames', []), rig.get('parents', [])
    offsets = np.asarray(rig.get('offsets'), dtype=float)
    if (not names or any(not isinstance(name, str) or not name for name in names) or
            len(set(names)) != len(names) or len(parents) != len(names) or
            offsets.shape != (len(names), 3) or not np.isfinite(offsets).all()):
        raise ValueError('Incomplete SOMA rig binding')
    if names[0] != 'Hips' or parents[0] != -1 or any(
            isinstance(parent, bool) or not isinstance(parent, int) or not 0 <= parent < joint
            for joint, parent in enumerate(parents[1:], 1)):
        raise ValueError('SOMA rig requires one Hips root and ordered parent bindings')
    lookup = {name: index for index, name in enumerate(names)}
    required = dict(Spine1='Hips', Spine2='Spine1', Chest='Spine2', Neck1='Chest', Neck2='Neck1', Head='Neck2')
    for side in ('Left', 'Right'):
        required.update({side+joint: side+parent if parent != 'Chest' and parent != 'Hips' else parent
                         for joint, parent in [('Shoulder','Chest'),('Arm','Shoulder'),('ForeArm','Arm'),
                             ('Hand','ForeArm'),('Leg','Hips'),('Shin','Leg'),('Foot','Shin'),
                             ('ToeBase','Foot'),('ToeEnd','ToeBase')]})
    for name, parent in required.items():
        if name not in lookup or names[parents[lookup[name]]] != parent:
            raise ValueError('Incomplete or incompatible SOMA chain: ' + name)
    rig_hash = hashlib.sha256(json.dumps(rig, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return lookup, rig_hash


def _bound_pose(motion, rig, rig_hash):
    validate_motion(motion, rig)
    if not math.isfinite(float(motion['fps'])):
        raise ValueError('Style references require a finite FPS')
    for key in ('joint_names', 'jointNames'):
        if key in motion and list(motion[key]) != rig['jointNames']:
            raise ValueError('Motion joint order differs from bound SOMA rig')
    if 'rigSha256' in motion and motion['rigSha256'] != rig_hash:
        raise ValueError('Motion rig digest differs from bound SOMA rig')
    rotations, positions = fk(motion['local_rot_mats'], motion['root_positions'], rig)
    for key, actual in [('global_rot_mats', rotations), ('posed_joints', positions)]:
        cached = motion.get(key)
        if cached is None or np.shape(cached) != actual.shape or not np.allclose(cached, actual, rtol=0, atol=1e-7):
            raise ValueError('Missing or incompatible complete pose/rig binding: ' + key)
    return rotations, positions


def _heading(rotations):
    forward = rotations[..., :, 2]
    if np.any(np.linalg.norm(forward[..., [0, 2]], axis=-1) < 1e-6):
        raise ValueError('Hips heading is undefined for a vertical forward axis')
    return np.arctan2(forward[..., 0], forward[..., 2])


def compose_upper_body_reference(native_reference, generated_style_motion, rig, *, fraction=.5):
    """Return (reference, receipt), preserving native Root/Hips/lower arrays.

    Sample one actual style pose, align its Hips yaw to each native frame, and
    transfer only Spine1 descendants' global orientations. Intended for sparse
    cycle-boundary fullbody anchors; it is not a generated asset or a style gate.
    Callers must bind the style discovery's executed generation receipt as well
    as the content digest returned here; arrays alone cannot prove model origin.
    """
    if isinstance(fraction, bool) or not isinstance(fraction, (int, float)) or not math.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError('Style reference fraction must be between zero and one')
    names, rig_hash = _rig_binding(rig)
    native_global, native_points = _bound_pose(native_reference, rig, rig_hash)
    style_global, _ = _bound_pose(generated_style_motion, rig, rig_hash)
    frame = round(fraction*(len(style_global)-1))
    yaw = _heading(native_global[:, names['Hips']]) - _heading(style_global[frame, names['Hips']])
    alignment = Rotation.from_euler('y', yaw[:, None]).as_matrix()
    upper = []
    for joint in range(len(rig['parents'])):
        ancestor = joint
        while ancestor >= 0 and ancestor != names['Spine1']:
            ancestor = rig['parents'][ancestor]
        if ancestor == names['Spine1']:
            upper.append(joint)
    lower = [joint for joint in range(len(rig['parents'])) if joint not in upper]
    local = native_reference['local_rot_mats'].copy()
    world = np.empty_like(native_global)
    for joint, parent in enumerate(rig['parents']):
        if joint in upper:
            desired = alignment @ style_global[frame, joint]
            local[:, joint] = world[:, parent].swapaxes(-1, -2) @ desired
        world[:, joint] = local[:, joint] if parent < 0 else world[:, parent] @ local[:, joint]
    result = complete(local, native_reference['root_positions'].copy(), rig, float(native_reference['fps']))
    if (not np.array_equal(result['local_rot_mats'][:, lower], native_reference['local_rot_mats'][:, lower]) or
            not np.array_equal(result['root_positions'], native_reference['root_positions'])):
        raise ValueError('Style reference composition changed native Root/Hips/lower body')
    original_feet, _ = foot_landmarks(native_reference, rig)
    composed_feet, _ = foot_landmarks(result, rig)
    foot_error = float(np.max(np.linalg.norm(composed_feet-original_feet, axis=-1)))
    lower_error = float(np.max(np.linalg.norm(result['posed_joints'][:, lower]-native_points[:, lower], axis=-1)))
    if foot_error > 1e-12 or lower_error > 1e-12:
        raise ValueError('Style reference composition changed native lower-body FK')
    content_hash = lambda value: _array_hash(value['local_rot_mats'], value['root_positions'], value['fps'])
    receipt = dict(schemaVersion=1, policy=POLICY, scope='optional_sparse_cycle_boundary_constraint_reference',
        productionReady=False, deliverable=False, referenceAcceptedAsMaster=False, styleQualityStatus='not_evaluated',
        nativeMotionSha256=content_hash(native_reference), styleMotionSha256=content_hash(generated_style_motion),
        outputMotionSha256=content_hash(result), rigSha256=rig_hash,
        styleFrame=frame, styleFraction=float(fraction), styleFrameCount=len(style_global),
        styleSampleTimeSeconds=frame/float(generated_style_motion['fps']), headingOwner='Hips',
        alignedHeadingDeltaRadiansRange=[float(yaw.min()), float(yaw.max())],
        selectedUpperJoints=[rig['jointNames'][joint] for joint in upper],
        modifiedJoints=[rig['jointNames'][joint] for joint in upper if
                        not np.array_equal(local[:, joint], native_reference['local_rot_mats'][:, joint])],
        preservedJoints=[rig['jointNames'][joint] for joint in lower],
        preservedRootAndLowerBody=True, maximumFootFkErrorMeters=foot_error, maximumLowerBodyFkErrorMeters=lower_error,
        requires=['executed_style_generation_receipt', 'native_phase_revalidation',
                  'actual_generated_authoring_gate', 'target_gait_gate', 'final_fbx_readback', 'runtime_gate'])
    return result, receipt

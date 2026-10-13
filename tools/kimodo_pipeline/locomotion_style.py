"""Idle-relative upper-body authoring, independent of locomotion contact repair."""
from __future__ import annotations

import copy
import math

import numpy as np
from scipy.spatial.transform import Rotation

from .gait import _array_hash
from .motion import complete, fk, validate_motion

POLICY = 'locomotion-idle-posture-v1'
BODY = ('Spine1', 'Spine2', 'Chest', 'Neck1', 'Neck2', 'Head')
ARMS = tuple(side + joint for side in ('Left', 'Right')
             for joint in ('Shoulder', 'Arm', 'ForeArm', 'Hand'))
LIMITS = dict(torsoForwardDegrees=8., neckForwardDegrees=8., headDownDegrees=8.,
              armAbductionDegrees=12.)


def validate_profile(profile):
    if profile.get('policy') == 'locomotion-idle-forearm-v2':
        from .locomotion_forearms import validate_profile as validate_forearms
        return validate_forearms(profile)
    expected = {'policy', 'bodyMotionGain', 'armMotionGain', 'referenceFraction', 'limits'}
    if set(profile) != expected or profile['policy'] != POLICY:
        raise ValueError('Unknown locomotion posture profile')
    for key in ('bodyMotionGain', 'armMotionGain', 'referenceFraction'):
        value = profile[key]
        if isinstance(value, bool) or not math.isfinite(value) or not 0 < value < 1:
            raise ValueError('Posture gains and reference fraction must be between zero and one')
    if set(profile['limits']) != set(LIMITS):
        raise ValueError('Unknown locomotion posture limits')
    if any(isinstance(v, bool) or not math.isfinite(v) or v <= 0 for v in profile['limits'].values()):
        raise ValueError('Posture limits must be finite and positive')


def _heading(rotations):
    forward = rotations[..., :, 2]
    return np.arctan2(forward[..., 0], forward[..., 2])


def author_posture(source, idle, rig, profile):
    """Attenuate upper-body deviations in the chest's heading, retaining actual swing.

    Root/pelvis and every leg local matrix are byte-identical to the input. Matching
    global upper rotations compensates pelvic tilt without changing leg support.
    This is an explicit offline authoring edit, not motion-model output or a gate.
    """
    if profile.get('policy') == 'locomotion-idle-forearm-v2':
        from .locomotion_forearms import author_posture as author_forearms
        return author_forearms(source, idle, rig, profile)
    validate_profile(profile)
    validate_motion(source, rig)
    validate_motion(idle, rig)
    names = {name: i for i, name in enumerate(rig['jointNames'])}
    if set(BODY + ARMS) - set(names):
        raise ValueError('Idle posture requires the complete SOMA upper-body chains')
    original, _ = fk(source['local_rot_mats'], source['root_positions'], rig)
    reference, _ = fk(idle['local_rot_mats'], idle['root_positions'], rig)
    frame = round(profile['referenceFraction'] * (len(reference) - 1))
    yaw = _heading(original[:, names['Chest']]) - _heading(reference[frame, names['Chest']])
    align = Rotation.from_euler('y', yaw[:, None]).as_matrix()
    local = source['local_rot_mats'].copy()
    global_out = np.empty_like(original)
    edited = set(BODY + ARMS)
    for i, parent in enumerate(rig['parents']):
        name = rig['jointNames'][i]
        if name in edited:
            baseline = align @ reference[frame, i]
            delta = baseline.swapaxes(-1, -2) @ original[:, i]
            gain = profile['bodyMotionGain'] if name in BODY else profile['armMotionGain']
            desired = baseline @ Rotation.from_rotvec(Rotation.from_matrix(delta).as_rotvec() * gain).as_matrix()
            local[:, i] = desired if parent < 0 else global_out[:, parent].swapaxes(-1, -2) @ desired
        global_out[:, i] = local[:, i] if parent < 0 else global_out[:, parent] @ local[:, i]
    untouched = [i for i, name in enumerate(rig['jointNames']) if name not in edited]
    if not np.array_equal(local[:, untouched], source['local_rot_mats'][:, untouched]):
        raise ValueError('Upper posture authoring changed an unrelated joint')
    result = complete(local, source['root_positions'].copy(), rig, float(source['fps']))
    report = dict(policy=POLICY, referenceFrame=frame, profile=profile,
                  inputMotionSha256=_array_hash(source['local_rot_mats'], source['root_positions'], source['fps']),
                  outputMotionSha256=_array_hash(result['local_rot_mats'], result['root_positions'], result['fps']),
                  idleMotionSha256=_array_hash(idle['local_rot_mats'], idle['root_positions'], idle['fps']),
                  preservedRootAndLowerBody=True, modifiedJoints=list(BODY + ARMS),
                  scope='Offline upper-body authoring; contact and artistic acceptance are separate')
    return result, report


def measure_posture(motion, rig):
    validate_motion(motion, rig)
    rotations, points = fk(motion['local_rot_mats'], motion['root_positions'], rig)
    names = {name: i for i, name in enumerate(rig['jointNames'])}
    heading = _heading(rotations[:, names['Chest']])
    forward = np.stack((np.sin(heading), np.zeros_like(heading), np.cos(heading)), axis=-1)
    lateral = np.stack((np.cos(heading), np.zeros_like(heading), -np.sin(heading)), axis=-1)
    def incline(a, b):
        segment = points[:, names[b]] - points[:, names[a]]
        return np.degrees(np.arctan2(np.sum(segment * forward, axis=-1), segment[:, 1]))
    head_forward = rotations[:, names['Head'], :, 2]
    result = dict(torsoForwardDegrees=incline('Spine1', 'Neck1'),
                  neckForwardDegrees=incline('Neck1', 'Head'),
                  headDownDegrees=np.degrees(np.arctan2(-head_forward[:, 1], np.linalg.norm(head_forward[:, [0, 2]], axis=-1))))
    for side in ('Left', 'Right'):
        segment = points[:, names[side + 'ForeArm']] - points[:, names[side + 'Arm']]
        result[side + 'ArmAbductionDegrees'] = np.degrees(np.arcsin(np.clip(
            np.abs(np.sum(segment * lateral, axis=-1)) / np.linalg.norm(segment, axis=-1), 0., 1.)))
    return {key: dict(median=float(np.median(value)), maximum=float(np.max(value))) for key, value in result.items()}


def audit_posture(motion, idle, rig, profile):
    if profile.get('policy') == 'locomotion-idle-forearm-v2':
        from .locomotion_forearms import audit_posture as audit_forearms
        return audit_forearms(motion, idle, rig, profile)
    validate_profile(profile)
    reference, observed = measure_posture(idle, rig), measure_posture(motion, rig)
    checks = []
    for key, value in observed.items():
        limit_key = 'armAbductionDegrees' if 'ArmAbduction' in key else key
        for statistic in ('median', 'maximum'):
            limit = reference[key][statistic] + profile['limits'][limit_key]
            checks.append(dict(metric=key, statistic=statistic, idle=reference[key][statistic],
                               observed=value[statistic], limit=limit, passed=value[statistic] <= limit))
    return dict(policy=POLICY, passed=all(row['passed'] for row in checks), checks=checks,
                profile=profile, idle=reference, observed=observed,
                motionSha256=_array_hash(motion['local_rot_mats'], motion['root_positions'], motion['fps']),
                idleMotionSha256=_array_hash(idle['local_rot_mats'], idle['root_positions'], idle['fps']),
                scope='Idle-relative posture only; does not certify contact or feminine artistic identity')


def targets(contracts):
    from .scope import is_prop
    families = {'gait_cycle', 'gait_pose', 'base_pose', 'sprint_start', 'acceleration_additive',
                'foot_transition', 'stance_transition', 'turn', 'rotate', 'idle_additive'}
    return [f"asset:{row['edition']}:{row['asset']}" for row in contracts['assets']
            if not is_prop(row) and (row['family'] in families
                or row['family'] == 'lean_additive' and not row['semanticName'].startswith('Air_')
                or row['family'] == 'pose_table' and row['semanticName'] in ('Default_Poses', 'Normal_Poses', 'Feminine_Poses'))]


def configure(nodes, contracts, profile):
    """Style actual generated candidates before existing cycle/contact gates."""
    from .graph import topological_order
    validate_profile(profile)
    selected = targets(contracts)
    order = topological_order(nodes, selected)
    idle = 'motion:Idle'
    v2 = profile['policy'] == 'locomotion-idle-forearm-v2'
    families = {row['semanticName']: row['family'] for row in contracts['assets']}
    def contextual(semantic):
        if v2:
            from .locomotion_forearms import context_profile
            return context_profile(profile, semantic, families.get(semantic))
        return copy.deepcopy(profile)
    if v2:
        original_idle='posture-input:'+idle
        nodes[original_idle]=copy.deepcopy(nodes[idle]);nodes[original_idle]['id']=original_idle
        nodes[idle]=dict(id=idle,op='idle_arm_posture',deps=['rig:soma',original_idle],
            params=dict(source=original_idle,profile=contextual('Idle')))
    candidates = [key for key in order if nodes[key]['op'] == 'generate'
                  and not key.endswith(':discovery')
                  and nodes[key]['params']['semantic'] not in ('Stand_Pose', 'Crouch_Pose', 'Idle')]
    for key in candidates:
        node = nodes[key]
        node['deps'] = list(dict.fromkeys([*node['deps'], idle]))
        candidate_profile = contextual(node['params']['semantic'])
        node['params'].update(postureIdleReference=idle, postureProfile=candidate_profile)
        styled = 'posture:' + key
        # Replace consumers before adding the new node, preserving its raw input.
        for consumer in list(nodes.values()):
            if key in consumer['deps']:
                consumer['deps'] = [styled if value == key else value for value in consumer['deps']]
                for field in ('source', 'gaitReference'):
                    if consumer['params'].get(field) == key:
                        consumer['params'][field] = styled
        nodes[styled] = dict(id=styled, op='idle_posture', deps=['rig:soma', key, idle],
                             params=dict(source=key, idle=idle, profile=candidate_profile))
    # V2 preserves crouching spine rotations instead of applying standing posture.
    key = 'motion:Crouch_Pose'
    original = 'posture-input:' + key
    nodes[original] = copy.deepcopy(nodes[key]); nodes[original]['id'] = original
    nodes[key] = dict(id=key, op='idle_posture', deps=['rig:soma', original, idle],
                      params=dict(source=original, idle=idle, profile=contextual('Crouch_Pose')))
    for target in selected:
        row = nodes[target]['params']['contract']
        if row['semanticName']=='Idle' and v2:
            nodes[target]['params']['contract']=dict(row,targetLocomotionPosturePolicy=profile['policy'])
            continue
        if row['semanticName'] in ('Idle', 'Stand_Pose'):
            continue
        source = nodes[target]['params']['source']
        # Table rows contain deliberately different stances; audit their upright upper body too.
        nodes[source]['deps'] = list(dict.fromkeys([*nodes[source]['deps'], idle]))
        nodes[source]['params'].update(postureValidationIdle=idle, postureProfile=contextual(row['semanticName']))
        nodes[target]['params']['contract'] = dict(row, targetLocomotionPosturePolicy=profile['policy'])


def validate_delivery(job):
    """Reject missing, altered, failed or unrelated posture approval."""
    if not job['contract'].get('targetLocomotionPosturePolicy'):
        return
    from .runner import file_hash
    from .graph import read_json
    policy = job['contract']['targetLocomotionPosturePolicy']
    if policy not in (POLICY, 'locomotion-idle-forearm-v2'):
        raise ValueError('Unknown delivered locomotion posture policy')
    receipt = job.get('locomotionPostureReceipt')
    if not isinstance(receipt, dict):
        raise ValueError('Missing locomotion posture receipt')
    for key, hash_key in (('file', 'sha256'), ('sourceMotionFile', 'sourceMotionSha256')):
        if not receipt.get(key) or file_hash(receipt[key]) != receipt.get(hash_key):
            raise ValueError('Changed locomotion posture evidence: ' + key)
    if receipt.get('deliveredMotionSha256') != job['motionSha256'] or file_hash(job['motionPath']) != job['motionSha256']:
        raise ValueError('Posture receipt does not bind delivered motion')
    report = read_json(receipt['file'])
    if report.get('policy') != policy or report.get('passed') is not True:
        raise ValueError('Upstream locomotion posture approval failed')
    with np.load(receipt['sourceMotionFile'], allow_pickle=False) as source:
        if report['motionSha256'] != _array_hash(source['local_rot_mats'], source['root_positions'], source['fps']):
            raise ValueError('Posture receipt belongs to a different motion')

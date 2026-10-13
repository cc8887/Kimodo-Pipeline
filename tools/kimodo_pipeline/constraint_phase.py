"""Verified native phase evidence for generation conditions, never acceptance.

A contact transition can be rejected by the SOMA authoring detector because its
centered derivative includes an airborne neighbour. Native target-qualified sole
intervals may still seed constraints. The generated result keeps every original
authoring, target and runtime gate.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from .als_reference import original_input
from .gait import audit_candidate, foot_landmarks
from .graph import digest, read_json


POLICY = 'native-constraint-phase-v1'
SCOPE = 'generation_constraint_phase_only'
MAXIMUM_TOE_MAPPING_ERROR_METERS = 1e-7


def _file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _runs(mask):
    edges = np.diff(np.r_[False, mask, False].astype(int))
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))


def build_native_phase_evidence(index_path, source, reference_motion, rig):
    """Recompute target qualification and bind its actual sole phases to this SOMA clip.

    Only an immutable native payload that passes the existing target gate and a
    projection preserving its toe trajectory may supply this evidence. No caller
    may supply contact masks, thresholds or a pre-approved target report.
    """
    from . import target_gait
    index_path = Path(index_path).resolve()
    index, row, payload_path = original_input(index_path, source)
    protocol = target_gait.original_protocol()
    # original_protocol is cached; verify its complete byte-bound evidence before
    # reuse, including contracts that choose the native stop/phase semantics.
    for relative, expected in protocol['sourceHashes'].items():
        path = (target_gait.ROOT/relative).resolve()
        if not path.is_relative_to(target_gait.ROOT.resolve()) or _file_hash(path) != expected:
            raise ValueError('Changed native phase protocol input: ' + relative)
    index_hash = _file_hash(index_path)
    if index_hash != protocol['sourceHashes']['assets/config/refactored_animation_sources.json']:
        raise ValueError('Phase index differs from immutable target protocol')
    payload = read_json(payload_path)
    raw = payload['raw']
    skeleton = index['skeletons']['/ALS/ALS/Character/SK_Als.SK_Als']
    native_gate = target_gait.analyze_target_gait(raw, skeleton, source, protocol)
    if native_gate.get('passed') is not True:
        raise ValueError('Native target gait must pass before supplying constraint phase evidence')
    name = target_gait.asset_name(source)
    speed_cm_s, kind = target_gait.PILOTS[name]
    original = protocol['assets'][kind]
    # Evidence belongs to this exact source rather than an alias sharing its gait kind.
    if original['source'] != source:
        raise ValueError('Constraint phase source differs from native protocol source')
    direction = np.asarray(target_gait.DIRECTIONS[name], dtype=float)
    count = int(raw['sampledKeyCount'])
    period = count - 1
    fps = period / float(raw['playLength'])
    frame_count = len(reference_motion['root_positions'])
    if not np.isclose(float(reference_motion['fps']), fps, atol=1e-9, rtol=0) or frame_count < 3*period+1:
        raise ValueError('Native phase reference requires matching FPS and three complete cycles')
    points, _velocity, _ = target_gait._sole_geometry(raw, skeleton, speed_cm_s, direction)
    frames = np.arange(frame_count)
    travel = frames[:, None] / fps * speed_cm_s * .01 * direction
    expected_toes = points[frames % period, :, 1][:, :, [0, 2, 1]] * .01
    expected_toes[:, :, 0] += travel[:, 0, None]
    expected_toes[:, :, 2] += travel[:, 1, None]
    actual, _ = foot_landmarks(reference_motion, rig)
    error = float(np.max(np.linalg.norm(actual[:, :, 1] - expected_toes, axis=-1)))
    if error > MAXIMUM_TOE_MAPPING_ERROR_METERS:
        raise ValueError(f'Native toe phase projection changed: {error:.9g}m')
    original_contacts = np.asarray(native_gate['contacts'], dtype=bool).T
    if original_contacts.shape != (count, 2):
        raise ValueError('Invalid qualified native contact layout')
    contacts = original_contacts[frames % period]
    if original['contactMode'] not in ('toe', 'flat_sole'):
        raise ValueError('Unsupported native phase contact geometry')
    sample_contacts = np.zeros((frame_count, 2, 2), dtype=bool)
    sample_contacts[:, :, 1] = contacts
    if original['contactMode'] == 'flat_sole':
        # Both native landmarks qualified together in analyze_target_gait.
        # Do not widen a forefoot interval into an invented heel contact.
        sample_contacts[:, :, 0] = contacts
    events = []
    for side, label in enumerate(('left', 'right')):
        for start, stop in _runs(contacts[:, side]):
            if start > 0:
                events.append(dict(frame=start, time=start/fps, side=label, kind='touchdown'))
            if stop < frame_count:
                events.append(dict(frame=stop, time=stop/fps, side=label, kind='liftoff'))
    events.sort(key=lambda event: (event['frame'], event['side'], event['kind']))
    starts = [event['frame'] for event in events if event['side'] == 'left' and event['kind'] == 'touchdown']
    cycles = []
    for start, end in zip(starts, starts[1:]):
        opposite = [event['frame'] for event in events if event['side'] == 'right' and
                    event['kind'] == 'touchdown' and start < event['frame'] < end]
        phase = (opposite[0]-start)/(end-start) if len(opposite) == 1 else None
        if end-start != period or phase is None or abs(phase-.5) > protocol['halfCycleTolerance']:
            raise ValueError('Native constraint phases do not form alternating full cycles')
        cycles.append(dict(start=start, endInclusive=end, durationSeconds=(end-start)/fps,
            oppositeTouchdowns=opposite, oppositePhase=phase, checks=[],
            scope=SCOPE, acceptedAsMaster=False))
    if len(cycles) < 2:
        raise ValueError('Native phase evidence requires two observed alternating cycles')
    audit = audit_candidate(reference_motion, rig, speed_cm_s*.01)
    evidence = dict(schemaVersion=1, policy=POLICY, scope=SCOPE, deliverable=False,
        productionReady=False, referenceAcceptedAsMaster=False,
        indexPath=str(index_path), indexSha256=index_hash, source=source,
        sourcePayloadPath=str(payload_path), sourcePayloadSha256=row['sha256'],
        referenceMotionSha256=audit['sourceMotionSha256'], rigSha256=audit['rigSha256'],
        sourceAuditPass=audit['pass'], sourceAuditSha256=digest(audit),
        nativeTargetGate=native_gate, nativeTargetGateSha256=digest(native_gate),
        targetPeriodSeconds=float(raw['playLength']), targetPeriodFrames=period,
        expectedSpeedMps=speed_cm_s*.01, travelDirectionXZ=direction.tolist(),
        referenceFrameCount=frame_count, fps=fps,
        maximumToeMappingErrorMeters=error, toeMappingToleranceMeters=MAXIMUM_TOE_MAPPING_ERROR_METERS,
        nativeContactMode=original['contactMode'],
        nativeStableFrames={label: np.flatnonzero(original_contacts[:-1, side]).tolist()
                            for side, label in enumerate(('left', 'right'))},
        detection=dict(contacts=contacts.astype(int).tolist(),
                       sampleContacts=sample_contacts.astype(int).tolist(), events=events), cycles=cycles,
        requires=['unchanged_generated_authoring_gate', 'target_gait_gate', 'final_fbx_readback', 'runtime_gate'])
    evidence['sha256'] = digest(evidence)
    return evidence


def validate_native_phase_evidence(evidence, reference_motion, rig):
    """Replay source/target/FK evidence rather than trusting passed flags or masks."""
    if not isinstance(evidence, dict) or evidence.get('policy') != POLICY or evidence.get('scope') != SCOPE:
        raise ValueError('Unsupported native constraint phase evidence')
    if (evidence.get('deliverable') is not False or evidence.get('productionReady') is not False or
            evidence.get('referenceAcceptedAsMaster') is not False):
        raise ValueError('Native phase evidence cannot approve a master or deliverable')
    if evidence.get('sha256') != digest({key: value for key, value in evidence.items() if key != 'sha256'}):
        raise ValueError('Changed native phase evidence digest')
    current = build_native_phase_evidence(evidence['indexPath'], evidence['source'], reference_motion, rig)
    if digest(current) != digest(evidence):
        raise ValueError('Native phase evidence does not match current payload, motion, rig or target gate')
    return current

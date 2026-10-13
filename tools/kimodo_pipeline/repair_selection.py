"""Deterministic bounded repair selection, driven by independent output failures.

Native phases can propose complete source spans. They cannot accept a repaired
performance: every returned motion is re-audited with the original gait policy.
"""
from __future__ import annotations

import copy
import heapq
import math

from . import gait
from .constraint_phase import validate_native_phase_evidence
from .periodic_repair import repair_periodic


POLICY = 'bounded-periodic-repair-selection-v1'
LIMITS = dict(max_pose_fit_degrees=20., max_ankle_correction=.15, max_hip_lowering=.05)
DEFAULT_OPTIONS = dict(ramp_frames=4, minimum_knee_bend_degrees=15.,
                       fit_regularization=.02, enforce_sole_clearance=True)
OPTION_KEYS = set(LIMITS) | set(DEFAULT_OPTIONS) | {
    'cycles', 'upper_body_harmonics', 'contact_phase_shift_frames',
    'upper_body_motion_scale', 'stabilize_knee_poles', 'flat_stance',
    'both_support_seams', 'stance_residual_lock', 'stance_residual_padding_frames'}


class RepairSelectionError(ValueError):
    def __init__(self, message, receipt):
        self.receipt = receipt
        super().__init__(message)


def _failure_classes(audit):
    checks = audit.get('checks', []) + [check for cycle in audit.get('cycles', [])
                                      for check in cycle.get('checks', [])]
    failures = sorted({check['name'] for check in checks if check.get('pass') is not True})
    classes = set()
    if any(name.startswith('seam') or name == 'cycle_contact_continuity' for name in failures):
        classes.add('seam')
    if any(name.endswith(('_geometric_slip', '_observed_support', '_near_ground_samples', '_penetration'))
           for name in failures):
        classes.add('contact')
    measurements = audit.get('detection', {}).get('measurements', {})
    policy = audit.get('policy', gait.DEFAULT_POLICY)
    def incomplete_contact(side):
        observed = measurements.get(side, {})
        median = observed.get('nearGroundMedianSpeedMps')
        return (observed.get('contactFrames') == 0 or
                observed.get('nearGroundSamples', math.inf) < policy['minimumNearGroundSamplesPerFoot'] or
                median is not None and median > policy['maximumNearGroundMedianSpeedMps'])
    if 'complete_cycles' in failures and any(incomplete_contact(side) for side in ('left', 'right')):
        classes.add('contact')
    if any(name in ('opposite_support_phase', 'cycle_duration', 'authored_speed') for name in failures):
        classes.add('phase-or-speed')
    if not classes and audit.get('pass') is not True:
        classes.add('unresolved-authoring')
    return failures, sorted(classes)


def _exception_class(error):
    if isinstance(error, OSError):
        return 'evidence-unavailable'
    text = str(error).lower()
    if 'unreachable' in text:
        return 'unreachable-anchor'
    if 'bound' in text or 'exceeds' in text:
        return 'correction-budget'
    if 'endpoint' in text or 'alternating cycle' in text:
        return 'cycle-evidence'
    return 'repair-error'


def select_periodic_repair(source, rig, speed, *, target_period_frames,
                           contact_reference=None, contact_phase_evidence=None,
                           repair_options=None, maximum_candidate_cycles=4,
                           maximum_attempts=24):
    """Return ``(motion, ordinary_repair_report + repairSelection)`` or reject.

    Try every retained complete span with harmonics 3/4 and no additional
    padding/smoothing first. Only actual contact/seam output failures schedule
    matching bounded retries. Stop on the first strict pass at the least
    invasive scheduled tier, not a claim of a global numerical optimum.
    ``RepairSelectionError.receipt`` preserves all executed attempts on failure.
    """
    receipt = dict(schemaVersion=1, policy=POLICY, scope='source_authoring_only',
                   productionReady=False, referenceAcceptedAsMaster=False,
                   targetPeriodFrames=target_period_frames, expectedSpeedMps=speed,
                   candidates=[], attempts=[], selectedAttempt=None)
    receipt['pass'] = False
    receipt['thresholds'] = dict(LIMITS)
    receipt['requestedRepairOptions'] = copy.deepcopy(repair_options)

    def reject(message):
        receipt['reason'] = message
        raise RepairSelectionError(message, receipt)

    if (not isinstance(target_period_frames, int) or isinstance(target_period_frames, bool)
            or target_period_frames < 10):
        reject('Bounded harmonics 3/4 require a target period of at least ten frames')
    if (not isinstance(maximum_candidate_cycles, int) or isinstance(maximum_candidate_cycles, bool)
            or not 1 <= maximum_candidate_cycles <= 8 or
            not isinstance(maximum_attempts, int) or isinstance(maximum_attempts, bool)
            or not 1 <= maximum_attempts <= 48):
        reject('Invalid bounded candidate or attempt budget')
    options = dict(DEFAULT_OPTIONS)
    if repair_options is not None:
        if not isinstance(repair_options, dict) or set(repair_options)-OPTION_KEYS:
            reject('Unknown or selector-owned periodic repair option')
        options.update(repair_options)
    for key, upper in LIMITS.items():
        value = options.setdefault(key, upper)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value <= upper:
            reject('Periodic repair cannot increase the original correction budget: '+key)
    receipt.update(thresholds={key: options[key] for key in LIMITS},
                   gaitPolicy=copy.deepcopy(gait.DEFAULT_POLICY), repairOptions=copy.deepcopy(options),
                   maximumCandidateCycles=maximum_candidate_cycles, maximumAttempts=maximum_attempts,
                   harmonics=[3, 4], contactPaddingFrames=[0, 1], seamSmoothingFrames=[0, 1, 2])
    try:
        audit = gait.audit_candidate(source, rig, speed)
        receipt.update(sourceMotionSha256=audit['sourceMotionSha256'], rigSha256=audit['rigSha256'],
                       sourceAudit=audit, sourceFrameCount=len(source['root_positions']), fps=float(source['fps']))
        verified_phase = None
        if contact_phase_evidence is not None:
            if contact_reference is None:
                reject('Native phase evidence requires its bound reference motion')
            verified_phase = validate_native_phase_evidence(contact_phase_evidence, contact_reference, rig)
            if (not math.isclose(verified_phase['expectedSpeedMps'], speed, abs_tol=1e-8, rel_tol=0) or
                not math.isclose(verified_phase['targetPeriodSeconds'], target_period_frames/float(source['fps']),
                                 abs_tol=1e-8, rel_tol=0)):
                reject('Native phase evidence differs from repair speed or target period')
            receipt['nativePhaseEvidence'] = dict(policy=verified_phase['policy'], sha256=verified_phase['sha256'],
                scope=verified_phase['scope'], referenceAcceptedAsMaster=False)
        reference = source if contact_reference is None else contact_reference
        reference_audit = audit if reference is source else gait.audit_candidate(reference, rig, speed)
        receipt['contactReferenceMotionSha256'] = reference_audit['sourceMotionSha256']
    except RepairSelectionError:
        raise
    except (ValueError, KeyError, TypeError, OSError) as error:
        receipt['preflightError'] = dict(errorType=type(error).__name__, error=str(error))
        reject('Repair selection input or native evidence was rejected')

    spans = []
    for cycle in audit['cycles']:
        if (len(cycle['oppositeTouchdowns']) == 1 and cycle['oppositePhase'] is not None and
                .35 <= cycle['oppositePhase'] <= .65 and .35 <= cycle['durationSeconds'] <= 2):
            spans.append((cycle['start'], cycle['endInclusive'], 'observed-generated-cycle'))
    if verified_phase is not None:
        # The verified constraint clock starts each complete generation cycle at
        # a multiple of its target period. This is a plan, never contact proof;
        # repair_periodic still checks actual generated endpoint geometry.
        spans += [(start, start+target_period_frames, 'native-phase-plan') for start in
                  range(0, len(source['root_positions'])-target_period_frames, target_period_frames)]
    # Prefer zero retiming; observed contacts break equal-duration ties. This
    # also prevents a noisy shorter observed cycle crowding verified full spans
    # out of the candidate budget before their geometry is checked.
    spans.sort(key=lambda span: (abs((span[1]-span[0])-target_period_frames),
                                span[2] != 'observed-generated-cycle', span[0], span[1]))
    unique = []
    seen_spans = set()
    for start, end, kind in spans:
        if (start, end) not in seen_spans:
            seen_spans.add((start, end))
            unique.append(dict(start=start, endInclusive=end, evidence=kind))
    receipt['completeCandidateCount'] = len(unique)
    receipt['candidateRanking'] = 'minimum retiming, measured contacts for equal periods, chronological source order'
    receipt['candidates'] = unique[:maximum_candidate_cycles]
    receipt['candidateLimitReached'] = len(unique) > maximum_candidate_cycles
    if not receipt['candidates']:
        reject('No complete observed or verified-phase source cycle is available')

    seen = set()
    retry_queue = []
    # Entries share no fitted state: every retry starts from the same source.
    def schedule(candidate_index, harmonics, padding, seam, cause):
        key = (candidate_index, harmonics, padding, seam)
        if key in seen:
            return
        seen.add(key)
        enabled = int(padding > 0)+int(seam > 0)
        heapq.heappush(retry_queue, (enabled, padding+seam, candidate_index, harmonics, padding, seam, cause))

    def attempt(candidate_index, harmonics, padding, seam, cause):
        candidate = receipt['candidates'][candidate_index]
        params = dict(options, target_period_frames=target_period_frames,
                      constrained_cycle_span=[candidate['start'], candidate['endInclusive']],
                      harmonics=harmonics, contact_padding_frames=padding, seam_smoothing_frames=seam)
        entry = dict(index=len(receipt['attempts']), candidateIndex=candidate_index,
                     parameters=params, trigger=cause)
        entry['pass'] = False
        receipt['attempts'].append(entry)
        try:
            repaired, report = repair_periodic(source, rig, speed, contact_reference=reference,
                contact_phase_evidence=contact_phase_evidence, **params)
            after = gait.audit_candidate(repaired, rig, speed)
            entry.update(after=after, outputMotionSha256=after['sourceMotionSha256'],
                repairEvidence={key: value for key, value in report.items() if key not in ('after', 'sourceAudit')})
            for report_key, limit_key in [('maximumPeriodicFitRotationDegrees', 'max_pose_fit_degrees'),
                    ('maximumAnkleCorrectionMeters', 'max_ankle_correction'), ('maximumHipLoweringMeters', 'max_hip_lowering')]:
                value = report[report_key]
                if not math.isfinite(value) or not 0 <= value <= options[limit_key]:
                    raise ValueError('Repair report exceeds original correction bound: '+report_key)
            failures, classes = _failure_classes(after)
            entry.update(failedChecks=failures, failureClasses=classes)
            entry['pass'] = after['pass'] is True
            if entry['pass']:
                receipt['pass'] = True
                receipt.update(selectedAttempt=entry['index'], attemptedCount=len(receipt['attempts']),
                    reason='First independent strict pass at the least invasive scheduled repair tier',
                    selectionIsGlobalCorrectionMinimum=False, attemptLimitReached=False)
                result = dict(report, after=after, repairAcceptedByAuthoringGate=True, repairSelection=receipt)
                return repaired, result
            if 'contact' in classes and padding == 0:
                schedule(candidate_index, harmonics, 1, seam, 'contact failure in attempt '+str(entry['index']))
            if 'seam' in classes and seam < 2:
                for radius in range(max(1, seam+1), 3):
                    schedule(candidate_index, harmonics, padding, radius, 'seam failure in attempt '+str(entry['index']))
        except (ValueError, ArithmeticError, OSError) as error:
            entry.update(errorType=type(error).__name__, error=str(error), failureClasses=[_exception_class(error)])
            if getattr(error, 'correction_budget', None) is not None:
                entry['correctionBudget'] = error.correction_budget
        return None

    ordinary = [(index, harmonics) for index in range(len(receipt['candidates'])) for harmonics in (3, 4)]
    for candidate_index, harmonics in ordinary:
        if len(receipt['attempts']) >= maximum_attempts:
            break
        result = attempt(candidate_index, harmonics, 0, 0, 'ordinary bounded periodic repair')
        if result is not None:
            return result
    while retry_queue and len(receipt['attempts']) < maximum_attempts:
        _, _, candidate_index, harmonics, padding, seam, cause = heapq.heappop(retry_queue)
        result = attempt(candidate_index, harmonics, padding, seam, cause)
        if result is not None:
            return result
    receipt.update(attemptedCount=len(receipt['attempts']),
                   attemptLimitReached=len(receipt['attempts']) >= maximum_attempts,
                   pendingRetries=len(retry_queue))
    reject('No bounded repair passed the unchanged independent authoring gate')

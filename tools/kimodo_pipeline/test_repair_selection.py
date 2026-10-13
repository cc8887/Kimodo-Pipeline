"""Failure-driven repair scheduling, with independent output acceptance."""
import copy
import unittest
from unittest.mock import patch

import numpy as np

from . import gait
from .repair_selection import RepairSelectionError, select_periodic_repair
from .test_gait import walking_fixture


class RepairSelectionTests(unittest.TestCase):
    def setUp(self):
        self.source, self.rig = walking_fixture(seconds=5.)
        self.audit = gait.audit_candidate(self.source, self.rig, .6)

    def report(self, source):
        return dict(after={'pass': True}, maximumPeriodicFitRotationDegrees=8.,
                    maximumAnkleCorrectionMeters=.04, maximumHipLoweringMeters=.01)

    def run_selection(self, **kwargs):
        return select_periodic_repair(self.source, self.rig, .6, target_period_frames=60, **kwargs)

    def tagged(self, failure):
        return dict(self.source, _test_failure=failure)

    def tagged_audit(self, source, rig, speed):
        result = copy.deepcopy(self.audit)
        if '_test_failure' not in source:
            return result
        name = source['_test_failure']
        result['pass'] = False
        result['selectedCycle'] = None
        result['checks'] = [dict(name='valid_complete_cycle', **{'pass': False})]
        for cycle in result['cycles']:
            cycle['pass'] = False
            cycle['checks'] = [dict(name=name, **{'pass': False})]
        return result

    def test_avoids_bad_cycle_without_raising_the_correction_budget(self):
        first = self.audit['cycles'][0]['start']
        def repair(source, rig, speed, **params):
            self.assertEqual(20., params['max_pose_fit_degrees'])
            self.assertEqual(.15, params['max_ankle_correction'])
            self.assertEqual(.05, params['max_hip_lowering'])
            if params['constrained_cycle_span'][0] == first:
                raise ValueError('Hip lowering exceeds bound: 0.1175m > 0.05')
            return source, self.report(source)
        with patch('tools.kimodo_pipeline.repair_selection.repair_periodic', side_effect=repair):
            result, report = self.run_selection()
        selection = report['repairSelection']
        self.assertTrue(selection['pass'])
        self.assertEqual(3, selection['attemptedCount'])
        self.assertEqual(1, selection['attempts'][-1]['candidateIndex'])
        self.assertEqual(['correction-budget'], selection['attempts'][0]['failureClasses'])
        self.assertIn('0.1175m', selection['attempts'][0]['error'])
        np.testing.assert_array_equal(result['root_positions'], self.source['root_positions'])

    def test_exhausted_budget_failures_are_not_a_success_or_cartesian_search(self):
        with patch('tools.kimodo_pipeline.repair_selection.repair_periodic',
                   side_effect=ValueError('Ankle correction exceeds bound: .19m')) as repair:
            with self.assertRaises(RepairSelectionError) as raised:
                self.run_selection()
        self.assertEqual(8, repair.call_count)
        report = raised.exception.receipt
        self.assertFalse(report['pass'])
        self.assertIsNone(report['selectedAttempt'])
        self.assertFalse(report['attemptLimitReached'])
        self.assertTrue(all(a['parameters']['contact_padding_frames'] == 0 and
                            a['parameters']['seam_smoothing_frames'] == 0 for a in report['attempts']))

    def test_later_unmodified_candidate_precedes_contact_or_seam_retries(self):
        first = self.audit['cycles'][0]['start']
        def repair(source, rig, speed, **params):
            result = self.tagged('seamVelocityMps') if params['constrained_cycle_span'][0] == first else source
            return result, self.report(result)
        with patch('tools.kimodo_pipeline.repair_selection.repair_periodic', side_effect=repair), \
                patch('tools.kimodo_pipeline.repair_selection.gait.audit_candidate', side_effect=self.tagged_audit):
            _, report = self.run_selection()
        attempts = report['repairSelection']['attempts']
        self.assertEqual(3, len(attempts))
        self.assertTrue(all(a['parameters']['seam_smoothing_frames'] == 0 for a in attempts))

    def test_only_the_failed_measurement_schedules_its_repair(self):
        for failure, option in [('seamVelocityMps', 'seam_smoothing_frames'),
                                ('left_geometric_slip', 'contact_padding_frames')]:
            with self.subTest(failure=failure):
                def repair(source, rig, speed, **params):
                    result = source if params[option] == 1 else self.tagged(failure)
                    return result, self.report(result)
                with patch('tools.kimodo_pipeline.repair_selection.repair_periodic', side_effect=repair), \
                        patch('tools.kimodo_pipeline.repair_selection.gait.audit_candidate', side_effect=self.tagged_audit):
                    _, report = self.run_selection(maximum_candidate_cycles=1)
                attempts = report['repairSelection']['attempts']
                self.assertEqual(3, len(attempts))
                other = 'contact_padding_frames' if option == 'seam_smoothing_frames' else 'seam_smoothing_frames'
                self.assertTrue(all(a['parameters'][other] == 0 for a in attempts))
                self.assertEqual(1, attempts[-1]['parameters'][option])

    def test_combined_retry_requires_failure_evidence_after_single_repair(self):
        def repair(source, rig, speed, **params):
            result = self.tagged('left_geometric_slip') if params['contact_padding_frames'] == 0 else \
                self.tagged('seamVelocityMps') if params['seam_smoothing_frames'] == 0 else source
            return result, self.report(result)
        with patch('tools.kimodo_pipeline.repair_selection.repair_periodic', side_effect=repair), \
                patch('tools.kimodo_pipeline.repair_selection.gait.audit_candidate', side_effect=self.tagged_audit):
            _, report = self.run_selection(maximum_candidate_cycles=1)
        attempts = report['repairSelection']['attempts']
        self.assertEqual(5, len(attempts))
        self.assertEqual(1, attempts[-1]['parameters']['contact_padding_frames'])
        self.assertEqual(1, attempts[-1]['parameters']['seam_smoothing_frames'])
        self.assertEqual('seam failure in attempt 2', attempts[-1]['trigger'])

    def test_sparse_contacts_do_not_hide_measured_slip_when_cycles_are_missing(self):
        def audit(source, rig, speed):
            result = copy.deepcopy(self.audit)
            if '_test_failure' in source:
                result.update(cycles=[], checks=[dict(name='complete_cycles', **{'pass': False})])
                result['pass'] = False
                for measurements in result['detection']['measurements'].values():
                    measurements.update(contactFrames=3, nearGroundMedianSpeedMps=.7)
            return result
        def repair(source, rig, speed, **params):
            result = source if params['contact_padding_frames'] == 1 else self.tagged('missing-cycle-slip')
            return result, self.report(result)
        with patch('tools.kimodo_pipeline.repair_selection.repair_periodic', side_effect=repair), \
                patch('tools.kimodo_pipeline.repair_selection.gait.audit_candidate', side_effect=audit):
            _, report = self.run_selection(maximum_candidate_cycles=1)
        self.assertEqual(3, report['repairSelection']['attemptedCount'])
        self.assertEqual(['contact'], report['repairSelection']['attempts'][0]['failureClasses'])

    def test_backend_pass_flag_cannot_approve_a_floating_result(self):
        floating = copy.deepcopy(self.source)
        floating['root_positions'][:, 1] += .3
        with patch('tools.kimodo_pipeline.repair_selection.repair_periodic',
                   return_value=(floating, self.report(floating))):
            with self.assertRaises(RepairSelectionError) as raised:
                self.run_selection(maximum_candidate_cycles=1)
        self.assertFalse(raised.exception.receipt['pass'])
        self.assertTrue(all(not a['after']['pass'] for a in raised.exception.receipt['attempts']))

    def test_attempt_limit_remains_a_rejection_with_pending_evidence(self):
        result = self.tagged('seamVelocityMps')
        with patch('tools.kimodo_pipeline.repair_selection.repair_periodic', return_value=(result, self.report(result))), \
                patch('tools.kimodo_pipeline.repair_selection.gait.audit_candidate', side_effect=self.tagged_audit):
            with self.assertRaises(RepairSelectionError) as raised:
                self.run_selection(maximum_attempts=1)
        self.assertFalse(raised.exception.receipt['pass'])
        self.assertTrue(raised.exception.receipt['attemptLimitReached'])
        self.assertEqual(2, raised.exception.receipt['pendingRetries'])

    def test_report_and_selection_are_deterministic(self):
        with patch('tools.kimodo_pipeline.repair_selection.repair_periodic',
                   return_value=(self.source, self.report(self.source))):
            a, first = self.run_selection()
            b, second = self.run_selection()
        self.assertEqual(first, second)
        np.testing.assert_array_equal(a['local_rot_mats'], b['local_rot_mats'])

    def test_original_limits_cannot_be_increased_or_hidden(self):
        with patch('tools.kimodo_pipeline.repair_selection.repair_periodic') as repair:
            with self.assertRaises(RepairSelectionError) as raised:
                self.run_selection(repair_options={'max_hip_lowering': .1175})
        repair.assert_not_called()
        self.assertEqual(.05, raised.exception.receipt['thresholds']['max_hip_lowering'])
        self.assertEqual(.1175, raised.exception.receipt['requestedRepairOptions']['max_hip_lowering'])
        with patch('tools.kimodo_pipeline.repair_selection.repair_periodic',
                   return_value=(self.source, dict(self.report(self.source), maximumAnkleCorrectionMeters=.151))):
            with self.assertRaises(RepairSelectionError) as raised:
                self.run_selection(maximum_candidate_cycles=1)
        self.assertTrue(all(a['failureClasses'] == ['correction-budget'] for a in raised.exception.receipt['attempts']))

    def test_native_phase_evidence_is_replayed_before_any_repair(self):
        with patch('tools.kimodo_pipeline.repair_selection.validate_native_phase_evidence',
                   side_effect=ValueError('Changed native phase evidence digest')), \
                patch('tools.kimodo_pipeline.repair_selection.repair_periodic') as repair:
            with self.assertRaises(RepairSelectionError) as raised:
                self.run_selection(contact_reference=self.source, contact_phase_evidence={'sha256': 'forged'})
        repair.assert_not_called()
        self.assertEqual([], raised.exception.receipt['attempts'])
        self.assertIn('Changed native phase', raised.exception.receipt['preflightError']['error'])

    def test_noisy_short_cycles_do_not_exclude_verified_full_spans(self):
        noisy = copy.deepcopy(self.audit)
        for cycle in noisy['cycles']:
            cycle['endInclusive'] -= 2
            cycle['durationSeconds'] = 58/60
        evidence = dict(expectedSpeedMps=.6, targetPeriodSeconds=1., policy='test-replayed',
                        sha256='test-bound-evidence', scope='generation_constraint_phase_only')
        with patch('tools.kimodo_pipeline.repair_selection.validate_native_phase_evidence', return_value=evidence), \
                patch('tools.kimodo_pipeline.repair_selection.gait.audit_candidate', side_effect=[noisy, self.audit]), \
                patch('tools.kimodo_pipeline.repair_selection.repair_periodic',
                      return_value=(self.source, self.report(self.source))) as repair:
            _, report = self.run_selection(contact_reference=self.source, contact_phase_evidence=evidence,
                                           maximum_candidate_cycles=1)
        self.assertEqual([0, 60], repair.call_args.kwargs['constrained_cycle_span'])
        self.assertEqual('native-phase-plan', report['repairSelection']['candidates'][0]['evidence'])

    def test_disappearing_bound_evidence_keeps_a_failed_receipt(self):
        with patch('tools.kimodo_pipeline.repair_selection.repair_periodic',
                   side_effect=FileNotFoundError('bound reference removed')):
            with self.assertRaises(RepairSelectionError) as raised:
                self.run_selection(maximum_candidate_cycles=1)
        self.assertFalse(raised.exception.receipt['pass'])
        self.assertEqual(2, raised.exception.receipt['attemptedCount'])
        self.assertTrue(all(a['failureClasses'] == ['evidence-unavailable'] for a in raised.exception.receipt['attempts']))

    def test_real_fixture_passes_original_gate_without_mutating_the_source(self):
        before = {name: value.copy() for name, value in self.source.items()}
        result, report = self.run_selection()
        self.assertTrue(gait.audit_candidate(result, self.rig, .6)['pass'])
        self.assertTrue(report['after']['pass'])
        self.assertLessEqual(report['maximumPeriodicFitRotationDegrees'], 20.)
        for name in before:
            np.testing.assert_array_equal(before[name], self.source[name])


if __name__ == '__main__':
    unittest.main()

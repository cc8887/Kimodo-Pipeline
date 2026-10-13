"""Native-only constraint phases never turn rejected source motions into masters."""
import copy
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

from .als_reference import native_gait_reference
from .constraint_phase import build_native_phase_evidence, validate_native_phase_evidence
from .gait import audit_candidate
from .graph import digest, read_json
from .paths import project_root, reference_root


class NativeConstraintPhaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Same real SOMA rig/reference environment as test_als_reference. Native
        # assets are intentionally external to Git and are not test downloads.
        receipts = read_json(project_root()/'artifacts/kimodo_als/nonprop-regeneration-20261009-v1/combined-receipts.json')
        cls.rig = read_json(receipts['rig:soma']['outputs']['rig.json']['path'])
        cls.index_path = reference_root()/'assets/config/refactored_animation_sources.json'
        index = read_json(cls.index_path)
        cls.source = next(row['source'] for row in index['assets'] if row['source'].endswith('.A_Als_Sprint'))
        cls.motion, cls.reference = native_gait_reference(cls.index_path, cls.source, cls.rig, 6., [0., 1.])
        cls.evidence = build_native_phase_evidence(cls.index_path, cls.source, cls.motion, cls.rig)

    def test_native_contact_evidence_does_not_approve_rejected_soma_reference(self):
        result = validate_native_phase_evidence(self.evidence, self.motion, self.rig)
        self.assertTrue(result['nativeTargetGate']['passed'])
        self.assertFalse(result['sourceAuditPass'])
        self.assertFalse(result['referenceAcceptedAsMaster'])
        self.assertFalse(result['deliverable'])
        self.assertEqual(result['nativeStableFrames'], {'left': [1, 2, 3], 'right': [10, 11, 12]})
        self.assertEqual(len(result['cycles']), 5)
        self.assertLess(result['maximumToeMappingErrorMeters'], 1e-12)
        self.assertFalse(audit_candidate(self.motion, self.rig, 6.)['pass'])
        samples = np.asarray(result['detection']['sampleContacts'])
        self.assertFalse(samples[:, :, 0].any())
        np.testing.assert_array_equal(samples[:, :, 1], result['detection']['contacts'])

    def test_failed_reference_audit_can_seed_feet_constraints_but_not_claim_acceptance(self):
        from .gait_constraints import materialize_gait_constraints, GaitConstraintError
        params = dict(seconds=3., fps=30, targetPeriodSeconds=.6,
                      rootPath=dict(kind='linear', velocity=6., direction=[0., 1.], headingDegrees=0),
                      poseConstraintMode='feet_only', maximumConstraintCycles=3,
                      preserveReferenceHeading=True, supportAnchorCount=3)
        with self.assertRaisesRegex(GaitConstraintError, 'no complete alternating'):
            materialize_gait_constraints(params, self.motion, self.rig)
        constraints, report = materialize_gait_constraints(params, self.motion, self.rig,
                                                          phase_evidence=self.evidence)
        self.assertTrue(report['pass'])
        self.assertFalse(report['referenceAuditPass'])
        self.assertFalse(report['referenceAcceptedAsMaster'])
        self.assertFalse(report['upperBodyConstrained'])
        self.assertEqual({'root2d', 'left-foot', 'right-foot'}, {x['type'] for x in constraints})
        self.assertLess(max(report['constraintCounts'].values()), 20)
        for key, value in [('velocity', 5.), ('direction', [0., -1.])]:
            bad = copy.deepcopy(params)
            bad['rootPath'][key] = value
            with self.assertRaisesRegex(GaitConstraintError, 'differs from requested'):
                materialize_gait_constraints(bad, self.motion, self.rig, phase_evidence=self.evidence)
        bad = copy.deepcopy(params)
        bad['targetPeriodSeconds'] = .8
        with self.assertRaisesRegex(GaitConstraintError, 'differs from requested'):
            materialize_gait_constraints(bad, self.motion, self.rig, phase_evidence=self.evidence)

    def test_modified_digest_or_rehashed_contacts_are_rejected(self):
        bad = copy.deepcopy(self.evidence)
        bad['detection']['contacts'][2] = [0, 0]
        with self.assertRaisesRegex(ValueError, 'digest'):
            validate_native_phase_evidence(bad, self.motion, self.rig)
        bad['sha256'] = digest({key: value for key, value in bad.items() if key != 'sha256'})
        with self.assertRaisesRegex(ValueError, 'does not match'):
            validate_native_phase_evidence(bad, self.motion, self.rig)

    def test_changed_motion_or_rig_cannot_borrow_native_phase_evidence(self):
        motion = copy.deepcopy(self.motion)
        motion['root_positions'][2, 1] += .01
        with self.assertRaisesRegex(ValueError, 'toe phase projection changed'):
            validate_native_phase_evidence(self.evidence, motion, self.rig)
        rig = copy.deepcopy(self.rig)
        rig['offsets'][rig['jointNames'].index('LeftToeEnd')][1] += .01
        with self.assertRaisesRegex(ValueError, 'toe phase projection changed'):
            validate_native_phase_evidence(self.evidence, self.motion, rig)

    def test_unchanged_toes_do_not_hide_changed_upper_body_from_motion_binding(self):
        motion = copy.deepcopy(self.motion)
        arm = self.rig['jointNames'].index('LeftArm')
        motion['local_rot_mats'][2, arm] = Rotation.from_euler('x', .1).as_matrix() @ motion['local_rot_mats'][2, arm]
        with self.assertRaisesRegex(ValueError, 'does not match'):
            validate_native_phase_evidence(self.evidence, motion, self.rig)

    def test_master_approval_cannot_be_added_to_phase_evidence(self):
        for flag in ('deliverable', 'productionReady', 'referenceAcceptedAsMaster'):
            bad = copy.deepcopy(self.evidence)
            bad[flag] = True
            bad['sha256'] = digest({key: value for key, value in bad.items() if key != 'sha256'})
            with self.assertRaisesRegex(ValueError, 'cannot approve'):
                validate_native_phase_evidence(bad, self.motion, self.rig)

    def test_failing_native_gate_is_not_replaced_by_saved_passed_flag(self):
        with patch('tools.kimodo_pipeline.target_gait.analyze_target_gait', return_value={'passed': False}):
            with self.assertRaisesRegex(ValueError, 'Native target gait must pass'):
                validate_native_phase_evidence(self.evidence, self.motion, self.rig)

    def test_forged_source_gate_approval_is_rejected_even_with_updated_digest(self):
        bad = copy.deepcopy(self.evidence)
        bad['sourceAuditPass'] = True
        bad['sha256'] = digest({key: value for key, value in bad.items() if key != 'sha256'})
        with self.assertRaisesRegex(ValueError, 'does not match'):
            validate_native_phase_evidence(bad, self.motion, self.rig)

    def test_wrong_fps_or_insufficient_cycle_context_is_rejected(self):
        motion = copy.deepcopy(self.motion)
        motion['fps'] = np.asarray(60.)
        with self.assertRaisesRegex(ValueError, 'matching FPS'):
            build_native_phase_evidence(self.index_path, self.source, motion, self.rig)
        short = {key: value[:19].copy() if isinstance(value, np.ndarray) and value.ndim else value
                 for key, value in self.motion.items()}
        with self.assertRaisesRegex(ValueError, 'three complete cycles'):
            build_native_phase_evidence(self.index_path, self.source, short, self.rig)

    def test_flat_sole_sample_contacts_match_qualified_native_interval_exactly(self):
        index = read_json(self.index_path)
        source = next(row['source'] for row in index['assets'] if row['source'].endswith('.A_Als_Walk_Backward'))
        motion, _ = native_gait_reference(self.index_path, source, self.rig, 1.5, [0., -1.])
        evidence = build_native_phase_evidence(self.index_path, source, motion, self.rig)
        self.assertEqual(evidence['nativeContactMode'], 'flat_sole')
        contacts = np.asarray(evidence['detection']['contacts'])
        samples = np.asarray(evidence['detection']['sampleContacts'])
        np.testing.assert_array_equal(samples[:, :, 0], contacts)
        np.testing.assert_array_equal(samples[:, :, 1], contacts)
        # A recomputed digest cannot authorize new contact outside native support.
        bad = copy.deepcopy(evidence)
        bad['detection']['sampleContacts'][0][0][0] = 1
        bad['sha256'] = digest({key: value for key, value in bad.items() if key != 'sha256'})
        with self.assertRaisesRegex(ValueError, 'does not match'):
            validate_native_phase_evidence(bad, motion, self.rig)


if __name__ == '__main__':
    unittest.main()

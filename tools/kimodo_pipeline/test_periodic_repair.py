"""Independent geometric checks for bounded periodic cleanup."""
import copy
import hashlib
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
from scipy.spatial.transform import Rotation

from .gait import foot_landmarks,audit_candidate
from .motion import complete, fk
from .periodic_repair import repair_periodic, _fade_stance_residual, _clear_pinned_sole, _CorrectionBudget
from .test_gait import walking_fixture


class NativeSoleCalibrationTests(unittest.TestCase):
    def test_bvh_export_basis_does_not_rotate_physical_rest_landmarks(self):
        source, rig = walking_fixture()
        rig = copy.deepcopy(rig)
        rig.pop('footLandmarks')
        rig['standardTPoseGlobalRotations'] = np.tile(
            Rotation.from_euler('x', 90, degrees=True).as_matrix(), (len(rig['jointNames']), 1, 1)).tolist()
        points, details = foot_landmarks(source, rig)
        rig['standardTPoseGlobalRotations'] = np.tile(np.eye(3), (len(rig['jointNames']), 1, 1)).tolist()
        other, _ = foot_landmarks(source, rig)
        np.testing.assert_allclose(points, other, atol=1e-12)
        self.assertLess(details['definitions']['left']['heel']['offset'][1], 0)


class PeriodicRepairTests(unittest.TestCase):
    def test_budget_keeps_frame_and_element_identity_before_taking_maximum(self):
        rig = dict(jointNames=['Hips', 'Leg'])
        rotation = np.array([[15., 0.], [0., 0.]])
        ankle = np.array([[.13, 0.], [0., 0.]])
        root = np.array([.04, 0.])
        budget = _CorrectionBudget(rig, 2, rotation, ankle, root)
        budget.add('seam', np.array([[0., 0.], [0., 15.]]),
                   np.array([[0., .04], [.04, 0.]]), np.array([0., .04]))
        legacy = dict(rotationDegrees=30., ankleMeters=.17, rootMeters=.08)
        limits = dict(rotationDegrees=20., ankleMeters=.15, rootMeters=.05)
        report = budget.enforce(limits, legacy, 'seam')
        self.assertEqual(15., report['peaks']['rotationDegrees']['value'])
        self.assertEqual(.13, report['peaks']['ankleMeters']['value'])
        self.assertEqual(.04, report['peaks']['rootMeters']['value'])
        self.assertEqual('Hips', report['peaks']['rotationDegrees']['bone'])
        self.assertEqual('left', report['peaks']['ankleMeters']['side'])
        self.assertEqual(0, report['peaks']['ankleMeters']['frame'])
        self.assertEqual(legacy, report['legacyConservativeSum'])
        # Identical contributions on the same frame/element must accumulate.
        budget.add('stance', rotation, ankle, root)
        with self.assertRaisesRegex(ValueError, 'Combined periodic and stance') as caught:
            budget.enforce(limits, legacy, 'stance')
        rejected = caught.exception.correction_budget
        self.assertEqual(30., rejected['peaks']['rotationDegrees']['value'])
        self.assertEqual(.26, rejected['peaks']['ankleMeters']['value'])
        self.assertEqual(.08, rejected['peaks']['rootMeters']['value'])

    def test_budget_counts_reversed_motion_as_two_correction_paths(self):
        source, rig = walking_fixture()
        before = complete(source['local_rot_mats'][:2], source['root_positions'][:2], rig, 60.)
        moved_root = before['root_positions'].copy()
        moved_root[:, 0] += .1
        moved_local = before['local_rot_mats'].copy()
        moved_local[:, 0] = Rotation.from_euler('y', 15., degrees=True).as_matrix() @ moved_local[:, 0]
        moved = complete(moved_local, moved_root, rig, 60.)
        budget = _CorrectionBudget(rig, 2, np.zeros((2, len(rig['jointNames']))),
                                   np.zeros((2, 2)), np.zeros(2))
        budget.add_motion('outward', before, moved)
        budget.add_motion('return', moved, before)
        report = budget.report({})
        self.assertAlmostEqual(.2, report['peaks']['rootMeters']['value'])
        self.assertAlmostEqual(30., report['peaks']['rotationDegrees']['value'])
        self.assertGreater(report['peaks']['ankleMeters']['value'], .15)

    def test_real_acceleration_exact_seam_budget_retains_independent_output_rejection(self):
        # Generated assets are deliberately outside Git. Run the numeric
        # regression when its independently hashed receipt fixture is present.
        from .motion import load_motion
        receipt_path = (Path(__file__).resolve().parents[2] /
                        'workspace/runs/generic-locomotion-20261013/swing-controls-receipts.json')
        if not receipt_path.is_file():
            self.skipTest('Receipt-verified generated SprintAccel fixture is not installed')
        receipts = json.loads(receipt_path.read_text(encoding='utf-8-sig'))
        def verified(entry):
            path = Path(entry['path'])
            self.assertEqual(entry['sha256'], hashlib.sha256(path.read_bytes()).hexdigest())
            return path
        rig = json.loads(verified(receipts['rig:soma']['outputs']['rig.json']).read_text(encoding='utf-8-sig'))
        source = load_motion(verified(receipts['candidate:Sprint_Acceleration:swing:neutral']['outputs']['motion.npz']), rig)
        outputs = receipts['reference:Sprint_Acceleration:native-als']['outputs']
        reference = load_motion(verified(outputs['motion.npz']), rig)
        evidence = json.loads(verified(outputs['native-phase-evidence.json']).read_text(encoding='utf-8-sig'))
        options = dict(constrained_cycle_span=[18, 36], harmonics=3, ramp_frames=4,
            minimum_knee_bend_degrees=15., fit_regularization=.02, enforce_sole_clearance=True,
            contact_padding_frames=1, target_period_frames=18, contact_reference=reference,
            contact_phase_evidence=evidence)
        repaired, report = repair_periodic(source, rig, 5., seam_smoothing_frames=1, **options)
        budget = report['correctionBudget']
        self.assertGreater(budget['legacyConservativeSum']['ankleMeters'], .15)
        self.assertAlmostEqual(.138655776198525, budget['peaks']['ankleMeters']['value'], places=8)
        self.assertAlmostEqual(13.106658593, budget['peaks']['rotationDegrees']['value'], places=6)
        self.assertFalse(report['after']['pass'])
        self.assertFalse(report['repairAcceptedByAuthoringGate'])
        independent = audit_candidate(repaired, rig, 5.)
        self.assertFalse(independent['pass'])
        self.assertGreater(independent['cycles'][0]['seamVelocityMps'], .75)
        with self.assertRaisesRegex(ValueError, 'Combined periodic and seam') as caught:
            repair_periodic(source, rig, 5., seam_smoothing_frames=2, **options)
        rejected = caught.exception.correction_budget
        self.assertAlmostEqual(28.4841, rejected['peaks']['rotationDegrees']['value'], places=3)
        self.assertAlmostEqual(.241950, rejected['peaks']['ankleMeters']['value'], places=5)
        self.assertEqual('RightShin', rejected['peaks']['rotationDegrees']['bone'])

    def test_forefoot_clearance_keeps_world_toe_planted_and_rejects_excessive_pitch(self):
        heel = np.array([0., -.08, -.05])
        toe = np.array([0., -.08, .16])
        anchor = np.array([.2, 0., 1.3])
        rotation = Rotation.from_euler('x', -15., degrees=True).as_matrix()
        ankle = anchor-rotation @ toe
        self.assertLess((ankle+rotation @ heel)[1], -.05)
        fixed, orientation, angle = _clear_pinned_sole(ankle, rotation, heel, toe, [False, True])
        np.testing.assert_allclose(fixed+orientation @ toe, anchor, atol=1e-12)
        self.assertGreaterEqual((fixed+orientation @ heel)[1], -1e-12)
        self.assertAlmostEqual(angle, 15.)
        rotation = Rotation.from_euler('x', -35., degrees=True).as_matrix()
        with self.assertRaisesRegex(ValueError, 'pitch correction exceeds bound'):
            _clear_pinned_sole(anchor-rotation @ toe, rotation, heel, toe, [False, True])
        # A heel landing uses the same pivot rule with the two ends swapped.
        rotation = Rotation.from_euler('x', 15., degrees=True).as_matrix()
        fixed, orientation, angle = _clear_pinned_sole(anchor-rotation @ heel, rotation, heel, toe, [True, False])
        np.testing.assert_allclose(fixed+orientation @ heel, anchor, atol=1e-12)
        self.assertGreaterEqual((fixed+orientation @ toe)[1], -1e-12)

    def verified_phase_fixture(self, upper_turn=20.):
        reference, rig = walking_fixture(seconds=5.)
        rig = copy.deepcopy(rig)
        rig['jointNames'] += ['Chest', 'RightForeArm', 'RightHandEnd']
        rig['parents'] += [0, 9, 10]
        rig['offsets'] += [[0, .2, 0], [.3, .1, 0], [.3, 0, 0]]
        local = np.concatenate([reference['local_rot_mats'],
            np.tile(np.eye(3), (len(reference['root_positions']), 3, 1, 1))], axis=1)
        reference = complete(local, reference['root_positions'], rig, 60.)
        audit = audit_candidate(reference, rig, .6)
        cycle = audit['cycles'][0]
        span = (cycle['start'], cycle['endInclusive'])
        local = local.copy()
        angles = (np.arange(len(local))-span[0])/(span[1]-span[0])*upper_turn
        local[:, 10] = Rotation.from_euler('z', angles[:, None], degrees=True).as_matrix()
        source = complete(local, reference['root_positions'].copy(), rig, 60.)
        # The native evidence validator has its own real-reference/hash tests.
        # This fixture isolates the order of repair prerequisites/output gates.
        evidence = dict(expectedSpeedMps=.6, targetPeriodSeconds=1., cycles=audit['cycles'],
                        detection=audit['detection'], policy='verified-test-phase', sha256='verified-fixture',
                        scope='generation_constraint_phase_only', sourceAuditPass=True)
        return source, reference, rig, span, evidence

    def test_verified_phase_allows_bounded_upper_body_closure_before_full_output_gate(self):
        source, reference, rig, span, evidence = self.verified_phase_fixture()
        with self.assertRaisesRegex(ValueError, 'endpoints do not match'):
            repair_periodic(source, rig, .6, harmonics=4,
                            constrained_cycle_span=span, contact_reference=reference)
        with patch('tools.kimodo_pipeline.constraint_phase.validate_native_phase_evidence', return_value=evidence) as validate:
            fixed, report = repair_periodic(source, rig, .6, harmonics=4,
                constrained_cycle_span=span, contact_reference=reference, contact_phase_evidence=evidence)
        validate.assert_called_once_with(evidence, reference, rig)
        self.assertGreater(report['sourceCycle']['seamBodyMeters'], .06)
        self.assertLessEqual(report['sourceCycle']['inputLowerBodySeamMeters'], .06)
        self.assertLessEqual(report['maximumPeriodicFitRotationDegrees'], 20.)
        self.assertTrue(report['after']['pass'], report['after']['checks'])
        self.assertTrue(audit_candidate(fixed, rig, .6)['pass'])

    def test_verified_phase_does_not_waive_pose_budget_root_or_leg_mismatch(self):
        source, reference, rig, span, evidence = self.verified_phase_fixture(160.)
        with patch('tools.kimodo_pipeline.constraint_phase.validate_native_phase_evidence', return_value=evidence):
            with self.assertRaisesRegex(ValueError, 'pose fitting exceeds correction bound'):
                repair_periodic(source, rig, .6, harmonics=4,
                    constrained_cycle_span=span, contact_reference=reference, contact_phase_evidence=evidence)
        source, reference, rig, span, evidence = self.verified_phase_fixture()
        damaged = copy.deepcopy(source)
        damaged['root_positions'][span[1], 1] += .1
        with patch('tools.kimodo_pipeline.constraint_phase.validate_native_phase_evidence', return_value=evidence):
            with self.assertRaisesRegex(ValueError, 'hip heights do not match'):
                repair_periodic(damaged, rig, .6, harmonics=4,
                    constrained_cycle_span=span, contact_reference=reference, contact_phase_evidence=evidence)
            damaged = copy.deepcopy(source)
            damaged['local_rot_mats'][span[1], 1] = Rotation.from_euler('x', 120, degrees=True).as_matrix()
            with self.assertRaisesRegex(ValueError, 'lower-body cycle endpoints'):
                repair_periodic(damaged, rig, .6, harmonics=4,
                    constrained_cycle_span=span, contact_reference=reference, contact_phase_evidence=evidence)

    def test_verified_phase_does_not_approve_a_failed_final_authoring_gate(self):
        source, reference, rig, span, evidence = self.verified_phase_fixture()
        source_audit = audit_candidate(source, rig, .6)
        reference_audit = audit_candidate(reference, rig, .6)
        rejected = copy.deepcopy(reference_audit)
        rejected['pass'] = False
        rejected['checks'].append({'name': 'independent_rejected_output', 'pass': False})
        with patch('tools.kimodo_pipeline.constraint_phase.validate_native_phase_evidence', return_value=evidence), \
             patch('tools.kimodo_pipeline.gait.audit_candidate', side_effect=[source_audit, reference_audit, rejected]):
            _, report = repair_periodic(source, rig, .6, harmonics=4,
                constrained_cycle_span=span, contact_reference=reference, contact_phase_evidence=evidence)
        self.assertFalse(report['repairAcceptedByAuthoringGate'])
        self.assertFalse(report['after']['pass'])

    def test_fast_swing_is_not_pulled_back_to_an_old_support_anchor(self):
        # The foot has already travelled 70 cm after toe-off. A support repair
        # of 2 cm must not acquire the whole swing displacement as correction.
        original = np.array([[0., .1, 0.], [0., .1, 0.], [0., .3, .7],
                             [0., .5, 1.], [0., .3, 1.4], [0., .1, 1.6]])
        stance = np.array([True, True, False, False, False, True])
        targets = original.copy()
        targets[stance, 0] += .02
        rotations = np.tile(np.eye(3), (len(original), 1, 1, 1))
        corrected_rotations = rotations.copy()
        corrected_rotations[stance, 0] = Rotation.from_euler('x', 10., degrees=True).as_matrix()
        result, rotation_result = _fade_stance_residual(
            original, rotations, targets, corrected_rotations, stance, 4)
        np.testing.assert_array_equal(result[stance], targets[stance])
        np.testing.assert_array_equal(rotation_result[stance], corrected_rotations[stance])
        np.testing.assert_array_equal(result[:, 1:], original[:, 1:])
        self.assertLessEqual(np.linalg.norm(result-original, axis=-1).max(), .02+1e-12)
        angles = np.rad2deg(np.linalg.norm(Rotation.from_matrix(rotation_result[:, 0]).as_rotvec(), axis=1))
        self.assertLessEqual(angles.max(), 10.+1e-12)
        # A perfect support correction is an identity over the entire swing.
        unchanged, unchanged_rot = _fade_stance_residual(
            original, rotations, original.copy(), rotations.copy(), stance, 4)
        np.testing.assert_array_equal(unchanged, original)
        np.testing.assert_array_equal(unchanged_rot, rotations)

    def test_stance_residual_lock_preserves_foot_rotation_and_bone_lengths_with_original_caps(self):
        source, rig = walking_fixture(seconds=5.)
        source['root_positions'][:, 0] += .01 * np.sin(np.arange(len(source['root_positions'])) / 60 * 2 * np.pi)
        original = copy.deepcopy(source)
        fixed, report = repair_periodic(source, rig, .6, harmonics=4, stance_residual_lock=True)
        self.assertTrue(report['after']['pass'])
        self.assertLessEqual(report['maximumPeriodicFitRotationDegrees'], 20.)
        self.assertLessEqual(report['maximumAnkleCorrectionMeters'], .15)
        self.assertLessEqual(report['maximumHipLoweringMeters'], .05)
        before, baseline = repair_periodic(source, rig, .6, harmonics=4)
        rotations_before, _ = fk(before['local_rot_mats'], before['root_positions'], rig)
        rotations_after, positions = fk(fixed['local_rot_mats'], fixed['root_positions'], rig)
        for bone in ('LeftFoot', 'RightFoot'):
            index = rig['jointNames'].index(bone)
            np.testing.assert_allclose(rotations_after[:, index], rotations_before[:, index], atol=1e-10)
        for joint, parent in enumerate(rig['parents']):
            if parent >= 0:
                np.testing.assert_allclose(np.linalg.norm(positions[:, joint] - positions[:, parent], axis=-1),
                                           np.linalg.norm(rig['offsets'][joint]), atol=1e-10)
        np.testing.assert_array_equal(source['local_rot_mats'], original['local_rot_mats'])
        np.testing.assert_array_equal(source['root_positions'], original['root_positions'])
        extra = report['stanceResidualLock']['maximumAdditionalRotationDegrees']
        self.assertGreater(extra, 0.)
        with self.assertRaisesRegex(ValueError, 'Combined periodic and stance'):
            repair_periodic(source, rig, .6, harmonics=4, stance_residual_lock=True,
                            max_pose_fit_degrees=baseline['maximumPeriodicFitRotationDegrees'] + extra / 2)
        with self.assertRaisesRegex(ValueError, 'padding exceeds'):
            repair_periodic(source, rig, .6, harmonics=4, stance_residual_lock=True, stance_residual_padding_frames=20)

    def test_both_measured_support_seams_remain_valid_after_mirroring(self):
        from .gait import author_cycle,mirror_authored_cycle
        from .motion import freeze
        source,rig=walking_fixture(seconds=5.)
        fixed,report=repair_periodic(source,rig,.6,harmonics=4,seam_smoothing_frames=1,both_support_seams=True)
        self.assertTrue(report['after']['pass'])
        self.assertEqual('left',report['seamSmoothing']['side'])
        self.assertEqual('right',report['seamSmoothing']['rightSeam']['side'])
        master,accepted=author_cycle(fixed,rig,expected_speed_mps=.6)
        _,mirrored=mirror_authored_cycle(master,freeze(source,0,1,rig),rig,accepted)
        self.assertTrue(mirrored['pass'])

    def test_seam_cleanup_preserves_stance_and_original_combined_budgets(self):
        source,rig=walking_fixture(seconds=5.)
        original=source['local_rot_mats'].copy()
        fixed,report=repair_periodic(source,rig,.6,harmonics=4,seam_smoothing_frames=1)
        self.assertTrue(report['after']['pass'])
        self.assertTrue(audit_candidate(fixed,rig,.6)['pass'])
        self.assertLessEqual(report['maximumPeriodicFitRotationDegrees'],20.)
        self.assertLessEqual(report['maximumAnkleCorrectionMeters'],.15)
        self.assertLessEqual(report['maximumHipLoweringMeters'],.05)
        _,points=fk(fixed['local_rot_mats'],fixed['root_positions'],rig)
        for joint,parent in enumerate(rig['parents']):
            if parent>=0:np.testing.assert_allclose(np.linalg.norm(points[:,joint]-points[:,parent],axis=-1),
                np.linalg.norm(rig['offsets'][joint]),atol=1e-10)
        np.testing.assert_array_equal(source['local_rot_mats'],original)
        _,baseline=repair_periodic(source,rig,.6,harmonics=4)
        combined_limit=baseline['maximumPeriodicFitRotationDegrees']+report['seamSmoothing']['maximumAdditionalRotationDegrees']/2
        with self.assertRaisesRegex(ValueError,'Combined periodic and seam'):
            repair_periodic(source,rig,.6,harmonics=4,seam_smoothing_frames=1,max_pose_fit_degrees=combined_limit)
        for radius in (True,-1,9):
            with self.assertRaises(ValueError):repair_periodic(source,rig,.6,harmonics=4,seam_smoothing_frames=radius)

    def test_bounded_contact_padding_is_a_repair_plan_not_acceptance(self):
        source,rig=walking_fixture(seconds=5.)
        original=source['local_rot_mats'].copy()
        fixed,report=repair_periodic(source,rig,.6,harmonics=4,contact_padding_frames=1)
        observed=np.asarray(report['measuredContactIntent'],bool)
        planned=np.asarray(report['repairContactIntent'],bool)
        np.testing.assert_array_equal(planned,observed|np.roll(observed,1,axis=0)|np.roll(observed,-1,axis=0))
        self.assertTrue(report['after']['pass'])
        self.assertTrue(audit_candidate(fixed,rig,.6)['pass'])
        np.testing.assert_array_equal(source['local_rot_mats'],original)
        with self.assertRaisesRegex(ValueError,'padding exceeds'):
            repair_periodic(source,rig,.6,harmonics=4,contact_padding_frames=20)

    def test_planned_reference_support_does_not_replace_measured_output_gate(self):
        reference,rig=walking_fixture(seconds=5.)
        source=copy.deepcopy(reference)
        source['root_positions'][:,0]+=.03*np.sin(np.arange(len(source['root_positions']))/60*2*np.pi)
        cycle=audit_candidate(reference,rig)['cycles'][0]
        span=(cycle['start'],cycle['endInclusive'])
        fixed,report=repair_periodic(source,rig,.6,harmonics=4,target_period_frames=60,
            constrained_cycle_span=span,contact_reference=reference,enforce_sole_clearance=True)
        self.assertTrue(report['after']['pass'])
        self.assertEqual('independently measured generation-reference heel/toe intervals',report['contactPlanSource'])
        self.assertLessEqual(report['maximumAnkleCorrectionMeters'],.15)
        self.assertLessEqual(report['maximumHipLoweringMeters'],.05)
        with self.assertRaisesRegex(ValueError,'correction exceeds bound'):
            repair_periodic(source,rig,.6,harmonics=4,constrained_cycle_span=span,
                contact_reference=reference,max_ankle_correction=1e-7)

    def test_constraint_window_requires_matching_output_pose_and_measured_contacts(self):
        source,rig=walking_fixture(seconds=5.)
        with self.assertRaisesRegex(ValueError,'independently measured'):
            repair_periodic(source,rig,.6,constrained_cycle_span=(60,120))
        damaged=copy.deepcopy(source);damaged['local_rot_mats'][120,1]=Rotation.from_euler('x',120,degrees=True).as_matrix()
        with self.assertRaisesRegex(ValueError,'endpoints do not match'):
            repair_periodic(damaged,rig,.6,constrained_cycle_span=(60,120),contact_reference=source)

    def test_period_change_repairs_stance_after_retiming(self):
        source, rig = walking_fixture(seconds=5.)
        repaired, report = repair_periodic(source, rig, .6, harmonics=4,target_period_frames=72)
        self.assertTrue(report['after']['pass'])
        self.assertEqual(72,report['outputPeriodFrames'])
        self.assertEqual(361,len(repaired['root_positions']))
        displacement=repaired['root_positions'][72]-repaired['root_positions'][0]
        self.assertAlmostEqual(.72,float(np.linalg.norm(displacement[[0,2]])),places=8)

    def test_cleanup_preserves_bone_lengths_and_periodic_motion(self):
        source, rig = walking_fixture(seconds=5.)
        root = source['root_positions'].copy()
        root[:, 0] += np.arange(len(root))/60*.08
        shifted = complete(source['local_rot_mats'], root, rig, 60)
        repaired, report = repair_periodic(shifted, rig, .6, harmonics=4)
        self.assertTrue(report['after']['pass'])
        _, positions = fk(repaired['local_rot_mats'], repaired['root_positions'], rig)
        for j, parent in enumerate(rig['parents']):
            if parent >= 0:
                np.testing.assert_allclose(np.linalg.norm(positions[:, j]-positions[:, parent], axis=-1),
                                           np.linalg.norm(rig['offsets'][j]), atol=1e-10)
        period = report['sourceCycle']['endInclusive']-report['sourceCycle']['start']
        np.testing.assert_allclose(repaired['local_rot_mats'][period:2*period],
                                   repaired['local_rot_mats'][2*period:3*period], atol=1e-9)
        self.assertFalse(report['productionReady'])

    def test_large_repair_does_not_silently_clip_or_stretch(self):
        source, rig = walking_fixture(seconds=5.)
        with self.assertRaisesRegex(ValueError, 'correction exceeds bound'):
            repair_periodic(source, rig, .6, max_ankle_correction=1e-7)


if __name__ == '__main__':
    unittest.main()

import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import gait, motion
from .loop_segment import crop_loop, tile_loop
from .loop_contact_solver import solve_loop_contacts
from .test_gait import walking_fixture


class JointContactTests(unittest.TestCase):
    def fixture(self):
        source, rig = walking_fixture(fps=30, seconds=5.)
        loop, report = crop_loop(source, rig, 30, 60, close=False)
        displacement = report['removedPlanarTrajectory']['displacement']
        detection = gait.detect_gait_events(tile_loop(loop, rig, displacement=displacement), rig)
        mask = np.asarray(detection['sampleContacts'][30:60], dtype=bool)
        return loop, rig, displacement, mask

    def test_healthy_motion_needs_only_phase_alignment_and_never_mutates_source(self):
        source, rig, displacement, mask = self.fixture()
        before = copy.deepcopy(source)
        result, report = solve_loop_contacts(source, rig, displacement, mask)
        self.assertTrue(report['passed'])
        self.assertFalse(report['applied'])
        for key in ('local_rot_mats', 'root_positions'):
            np.testing.assert_array_equal(source[key], before[key])
            np.testing.assert_array_equal(result[key], source[key][report['sourceFrameIndices']])
        self.assertTrue(gait.audit_candidate(tile_loop(result, rig, displacement=displacement), rig, .6)['pass'])

    def test_missing_bilateral_or_ambiguous_support_refused(self):
        source, rig, displacement, mask = self.fixture()
        for invalid in (np.zeros_like(mask), np.ones_like(mask), mask[:, :1], mask.astype(float)+.2):
            with self.subTest(shape=invalid.shape), self.assertRaises(ValueError):
                solve_loop_contacts(source, rig, displacement, invalid)

    def test_open_endpoints_and_nonplanar_travel_refused(self):
        source, rig, displacement, mask = self.fixture()
        with self.assertRaises(ValueError):
            solve_loop_contacts(source, rig, [0, .01, .6], mask)
        source['root_positions'][-1, 0] += .1
        with self.assertRaisesRegex(ValueError, 'closed input'):
            solve_loop_contacts(source, rig, displacement, mask)

    def test_optimizer_weights_are_bounded_and_cannot_change_gate_limits(self):
        source, rig, displacement, mask = self.fixture()
        for key in ('seam_weight', 'support_weight', 'near_ground_weight'):
            for value in (0., 11., float('nan')):
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    solve_loop_contacts(source, rig, displacement, mask, **{key: value})
        _, report = solve_loop_contacts(source, rig, displacement, mask, seam_weight=3., support_weight=2.)
        self.assertEqual(report['limits'], dict(rotationDegrees=20., ankleMeters=.15, rootMeters=.05))

    def test_actual_joint_solve_preserves_lengths_and_measures_final_budget(self):
        source, rig, displacement, mask = self.fixture()
        source['root_positions'][12, 0] += .05
        source = motion.complete(source['local_rot_mats'], source['root_positions'], rig, source['fps'])
        before = copy.deepcopy(source)
        self.assertFalse(gait.audit_candidate(tile_loop(source, rig, displacement=displacement), rig)['pass'])
        result, report = solve_loop_contacts(source, rig, displacement, mask, max_evaluations=5,
                                            optimize_shared_anchor=True, optimize_pelvis=True)
        self.assertTrue(report['applied'])
        for key in ('local_rot_mats', 'root_positions'):
            np.testing.assert_array_equal(source[key], before[key])
            np.testing.assert_array_equal(result[key][0], result[key][-1])
        joints = motion.fk(result['local_rot_mats'], result['root_positions'], rig)[1]
        for child, parent in enumerate(rig['parents']):
            if parent >= 0:
                np.testing.assert_allclose(np.linalg.norm(joints[:, child]-joints[:, parent], axis=-1),
                                           np.linalg.norm(rig['offsets'][child]), atol=1e-10)
        from .loop_seam import _paths
        original = motion.complete(before['local_rot_mats'][report['sourceFrameIndices']],
                                   before['root_positions'][report['sourceFrameIndices']], rig, before['fps'])
        changes = _paths(original, result, rig)
        for key, limit in report['limits'].items():
            self.assertGreaterEqual(report['cumulativePeaks'][key]+1e-10, changes[key].max())
            if report['passed']:
                self.assertLessEqual(report['cumulativePeaks'][key], limit)
        self.assertEqual(gait.audit_candidate(tile_loop(result, rig, displacement=displacement), rig, .6)['pass'],
                         report['after']['pass'])

    def test_original_budget_and_malformed_history_refused_before_optimizer(self):
        source, rig, displacement, mask = self.fixture()
        prior = dict(rotationDegrees=np.full(source['local_rot_mats'].shape[:2], 20.),
            ankleMeters=np.zeros((31, 2)), rootMeters=np.zeros(31))
        with self.assertRaisesRegex(ValueError, 'original correction budgets'):
            solve_loop_contacts(source, rig, displacement, mask, prior_paths=prior)
        for bad in ({}, dict(prior, rootMeters=[0]), dict(prior, rootMeters=np.full(31, np.nan))):
            with self.assertRaises(ValueError):
                solve_loop_contacts(source, rig, displacement, mask, prior_paths=bad)

    def test_posture_is_not_rebuilt_after_loop_acceptance(self):
        from .runner import Runner, write_json, read_json
        from .batch import run_batch
        source, rig = walking_fixture(seconds=5.)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_json(root/'rig.json', rig)
            motion.save_motion(root/'source.npz', source, rig)
            profile = {'policy': 'locomotion-idle-forearm-v2'}
            nodes = {'rig:soma': dict(id='rig:soma', op='rig', deps=[], params={}),
                'source': dict(id='source', op='external', deps=['rig:soma'], params={'reason': 'fixture'}),
                'crop': dict(id='crop', op='loop_crop', deps=['rig:soma', 'source'], params=dict(
                    source='source', startFrame=60, endFrameInclusive=120, expectedSpeedMps=.6,
                    postureValidationIdle='source', postureProfile=profile))}
            runner = Runner(dict(nodes=nodes, rules={'fps': 60}, rulesDigest='test'), root/'run', bindings={
                'rig:soma': {'path': str(root/'rig.json')}, 'source': {'path': str(root/'source.npz')}})
            with patch('tools.kimodo_pipeline.locomotion_style.author_posture', side_effect=lambda m, *args: (m, {})) as author:
                with patch('tools.kimodo_pipeline.locomotion_style.audit_posture', return_value={'passed': True}):
                    result = run_batch(runner, ['crop'])
            self.assertEqual(1, result['completedTargets'])
            self.assertEqual(1, author.call_count)
            saved = runner.motion('crop')
            self.assertTrue(gait.audit_candidate(saved, rig, .6)['pass'])
            self.assertTrue(read_json(runner.output_of('crop')['loop-crop.json']['path'])['playbackAuthoringAudit']['pass'])


if __name__ == '__main__':
    unittest.main()

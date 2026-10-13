"""Final seam repair must fix motion, preserve anchors and refuse bad budgets."""
import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from . import gait, motion
from .loop_segment import crop_loop, tile_loop
from .loop_seam import repair_loop_seam, interpolation_seam
from . import test_loop_segment


class FinalSeamTests(unittest.TestCase):
    def fixture(self):
        source, rig = test_loop_segment.LoopSegmentTests().fixture()
        loop, _ = crop_loop(source, rig, 15, 44, close=False)
        # A real directional velocity kink, with bounded pose mismatch.
        loop['local_rot_mats'][1, 0] = Rotation.from_euler('z', 9, degrees=True).as_matrix()
        return motion.complete(loop['local_rot_mats'], loop['root_positions'], rig, 30), rig

    def test_fixed_last_pose_and_middle_unchanged_after_actual_kink(self):
        source, rig = self.fixture()
        original = copy.deepcopy(source)
        self.assertGreater(interpolation_seam(source, rig)['velocityMps'], .75)
        result, report = repair_loop_seam(source, rig)
        self.assertTrue(report['passed'])
        for key in ('local_rot_mats', 'root_positions'):
            np.testing.assert_array_equal(source[key], original[key])
            np.testing.assert_array_equal(result[key][-1], source[key][-1])
            np.testing.assert_array_equal(result[key][0], source[key][-1])
            width = report['windowFrames']
            np.testing.assert_array_equal(result[key][width+1:-width-1], source[key][width+1:-width-1])
        self.assertLessEqual(report['after']['discrete']['velocityMps'], .75)
        self.assertLessEqual(report['after']['discrete']['angularVelocityRps'], 6.)
        for method in ('slerp', 'nlerp'):
            self.assertLessEqual(interpolation_seam(result, rig, method)['velocityMps'], .75)
        self.assertFalse(report['productionReady'])

    def test_first_anchor_and_rigid_rotation_invariance(self):
        source, rig = self.fixture()
        out, report = repair_loop_seam(source, rig, anchor='first')
        np.testing.assert_array_equal(out['local_rot_mats'][0], source['local_rot_mats'][0])
        rotated = copy.deepcopy(source)
        turn = Rotation.from_euler('y', 70, degrees=True).as_matrix()
        rotated['local_rot_mats'][:, 0] = turn @ source['local_rot_mats'][:, 0]
        rotated['root_positions'] = source['root_positions'] @ turn.T
        rotated = motion.complete(rotated['local_rot_mats'], rotated['root_positions'], rig, 30)
        _, rotated_report = repair_loop_seam(rotated, rig, anchor='first')
        self.assertAlmostEqual(report['after']['discrete']['velocityMps'],
                               rotated_report['after']['discrete']['velocityMps'], places=10)

    def test_bad_cumulative_budget_cannot_cancel_against_new_edit(self):
        source, rig = self.fixture()
        prior = dict(rotationDegrees=np.full(source['local_rot_mats'].shape[:2], 20.),
                     ankleMeters=np.zeros((len(source['root_positions']), 2)),
                     rootMeters=np.zeros(len(source['root_positions'])))
        with self.assertRaisesRegex(ValueError, 'original seam and correction limits') as context:
            repair_loop_seam(source, rig, prior_paths=prior)
        self.assertFalse(context.exception.seam_report['passed'])
        self.assertTrue(all(not item['budgetPass'] for item in context.exception.seam_report['attempts']))

    def test_cannot_close_an_excessive_pose_gap(self):
        source, rig = self.fixture()
        source['local_rot_mats'][0, 0] = Rotation.from_euler('z', 90, degrees=True).as_matrix()
        with self.assertRaisesRegex(ValueError, 'original seam and correction limits'):
            repair_loop_seam(source, rig)

    def test_zero_repair_is_idempotent(self):
        source, rig = self.fixture()
        result, _ = repair_loop_seam(source, rig)
        repeated, report = repair_loop_seam(result, rig)
        self.assertFalse(report['applied'])
        np.testing.assert_array_equal(repeated['local_rot_mats'], result['local_rot_mats'])

    def test_closed_seam_cannot_approve_grounded_sliding(self):
        from .test_gait import walking_fixture
        source, rig = walking_fixture(seconds=5.)
        source['root_positions'][:, 0] += np.arange(len(source['root_positions']))/60*.5
        loop, crop = crop_loop(source, rig, 60, 120, close=False)
        result, report = repair_loop_seam(loop, rig)
        self.assertTrue(report['passed'])
        audit = gait.audit_candidate(tile_loop(result, rig, displacement=crop['removedPlanarTrajectory']['displacement']), rig)
        self.assertFalse(audit['pass'])

    def test_raw_angular_failure_is_repaired_before_full_playback_gate(self):
        from .test_gait import walking_fixture
        source, rig = walking_fixture(seconds=5.)
        loop, crop = crop_loop(source, rig, 60, 120, close=False)
        toe = rig['jointNames'].index('RightToeEnd')
        loop['local_rot_mats'][1, toe] = Rotation.from_euler('x', 10, degrees=True).as_matrix()
        result, report = repair_loop_seam(loop, rig,
            playback_displacement=crop['removedPlanarTrajectory']['displacement'], expected_speed_mps=.6)
        self.assertFalse(report['before']['passSeam'])
        self.assertTrue(report['after']['passSeam'])
        self.assertTrue(report['playbackGatePass'])
        self.assertTrue(gait.audit_candidate(tile_loop(result, rig,
            displacement=crop['removedPlanarTrajectory']['displacement']), rig, .6)['pass'])

    def test_contact_failure_exhausts_alternatives_and_remains_diagnostic(self):
        source, rig = self.fixture()
        _, report = repair_loop_seam(source, rig, playback_displacement=[0, 0, .6], expected_speed_mps=.6)
        self.assertTrue(report['passed'])
        self.assertFalse(report['playbackGatePass'])
        self.assertFalse(report['productionReady'])
        eligible = [a for a in report['attempts'] if a['passSeam'] and a['budgetPass']]
        self.assertGreater(len(eligible), 1)
        self.assertTrue(all(not a['playbackGate']['passed'] for a in eligible))

    def test_runner_keeps_raw_failure_and_accepts_only_repaired_output(self):
        from .test_gait import walking_fixture
        from .runner import Runner, write_json, read_json
        from .batch import run_batch
        source, rig = walking_fixture(seconds=5.)
        toe = rig['jointNames'].index('RightToeEnd')
        source['local_rot_mats'][61, toe] = Rotation.from_euler('x', 10, degrees=True).as_matrix()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_json(root/'rig.json', rig)
            motion.save_motion(root/'source.npz', source, rig)
            nodes = {'rig:soma': dict(id='rig:soma', op='rig', deps=[], params={}),
                     'source': dict(id='source', op='external', deps=['rig:soma'], params={'reason': 'fixture'}),
                     'crop': dict(id='crop', op='loop_crop', deps=['rig:soma', 'source'], params=dict(
                         source='source', startFrame=60, endFrameInclusive=120, expectedSpeedMps=.6))}
            runner = Runner(dict(nodes=nodes, rules={'fps': 60}, rulesDigest='test'), root/'run', bindings={
                'rig:soma': {'path': str(root/'rig.json')}, 'source': {'path': str(root/'source.npz')}})
            result = run_batch(runner, ['crop'])
            self.assertEqual(1, result['completedTargets'])
            report = read_json(runner.output_of('crop')['loop-crop.json']['path'])
            self.assertFalse(report['finalSeamRepair']['before']['passSeam'])
            self.assertTrue(report['finalSeamRepair']['after']['passSeam'])
            self.assertTrue(report['playbackAuthoringAudit']['pass'])
            self.assertIn('pre-seam-loop.npz', runner.output_of('crop'))

    def test_invalid_windows_and_prior_paths_refused(self):
        source, rig = self.fixture()
        for width in (True, 1, 14, 15):
            with self.subTest(width=width), self.assertRaises(ValueError):
                repair_loop_seam(source, rig, window_frames=width)
        with self.assertRaises(ValueError):
            repair_loop_seam(source, rig, anchor='middle')
        with self.assertRaises(ValueError):
            interpolation_seam(source, rig, 'cubic')


if __name__ == '__main__':
    unittest.main()

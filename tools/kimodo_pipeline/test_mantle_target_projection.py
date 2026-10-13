"""Source/target feasibility without widening repair budgets or replacing style."""
import copy
from pathlib import Path
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from .gait_constraints import solve_foot_ik
from .graph import read_json
from .mantle import _arm_rig, reference_input
from .mantle_target_projection import TargetPoseObserver, TargetProjectionError, project_target_pose
from .motion import fk, load_motion, resample
from .offline_retarget import BASIS, mapping
from .publish import world_pose
from .test_mantle_constraint_feedback import rig_fixture


def controlled_fixture():
    rig = rig_fixture()
    for side in ('Left', 'Right'):
        rig['parents'].append(rig['jointNames'].index(side+'Foot'))
        rig['jointNames'].append(side+'ToeBase')
        rig['offsets'].append([0., 0., .12])
    reverse = {source: target for target, source in mapping().items()}
    kept = [index for index, name in enumerate(rig['jointNames']) if name in reverse]
    index_map = {source: target+1 for target, source in enumerate(kept)}
    names = ['root']+[reverse[rig['jointNames'][index]] for index in kept]
    parents = [-1]+[index_map.get(rig['parents'][index], 0) for index in kept]
    reference = [dict(position=[0, 0, 0], rotation=[0, 0, 0, 1], scale=[1, 1, 1])]
    for index in kept:
        offset = BASIS @ np.asarray(rig['offsets'][index])*100
        if rig['jointNames'][index] in ('Spine1', 'Spine2', 'Chest'):
            offset *= .65
        reference.append(dict(position=offset.tolist(), rotation=[0, 0, 0, 1], scale=[1, 1, 1]))
    skeleton = dict(source='controlled-target', rawBoneNames=names, rawParents=parents, referencePose=reference)
    local = np.repeat(np.eye(3)[None], len(rig['jointNames']), axis=0)
    local[rig['jointNames'].index('LeftForeArm')] = Rotation.from_euler('y', 55, degrees=True).as_matrix()
    local[rig['jointNames'].index('LeftFinger')] = Rotation.from_euler('z', 21, degrees=True).as_matrix()
    root = np.array([0., .8, 0.])
    return rig, skeleton, local, root


class MantleTargetProjectionTests(unittest.TestCase):
    def setUp(self):
        self.rig, self.skeleton, self.local, self.root = controlled_fixture()
        self.observer = TargetPoseObserver(self.rig, self.skeleton)
        rot, pos = fk(self.local[None], self.root[None], self.rig)
        hand = self.rig['jointNames'].index('LeftHand')
        self.source = [dict(side='left', kind='hand', position=pos[0, hand], rotation=rot[0, hand])]
        target_rot, target_pos = self.observer.observe(self.local, self.root)
        hand = self.observer.lookup['hand_l']
        self.target = [dict(side='left', position=target_pos[hand], rotation=target_rot[hand])]

    def project(self, **kwargs):
        return project_target_pose(self.local, self.root, self.rig, self.source, self.root,
                                   self.skeleton, self.target, observer=self.observer, **kwargs)

    def test_already_feasible_pose_preserves_root_free_joints_and_lengths(self):
        old_local = self.local.copy(); old_rig = copy.deepcopy(self.rig)
        result, root, receipt = self.project()
        self.assertTrue(receipt['passed'])
        self.assertFalse(receipt['deliverable'])
        np.testing.assert_array_equal(root, self.root)
        self.assertEqual(self.rig, old_rig)
        np.testing.assert_array_equal(self.local, old_local)
        np.testing.assert_array_equal(result[self.rig['jointNames'].index('LeftFinger')],
                                      self.local[self.rig['jointNames'].index('LeftFinger')])
        _, points = fk(result[None], root[None], self.rig)
        for index, parent in enumerate(self.rig['parents']):
            if parent >= 0:
                self.assertAlmostEqual(np.linalg.norm(points[0, index]-points[0, parent]),
                                       np.linalg.norm(self.rig['offsets'][index]), places=10)

    def test_target_proportion_difference_is_reduced_while_source_wrist_stays_fixed(self):
        # A known feasible alternate torso pose defines a target, without
        # handing the solver that pose or any reference full-body rotations.
        feasible = self.local.copy()
        feasible[0] = Rotation.from_euler('x', 10, degrees=True).as_matrix()
        feasible = solve_foot_ik(feasible, self.root, _arm_rig(self.rig), 'left',
                                self.source[0]['position'], self.source[0]['rotation'], {})
        rotations, points = self.observer.observe(feasible, self.root)
        hand = self.observer.lookup['hand_l']
        self.target[0].update(position=points[hand], rotation=rotations[hand])
        result, root, receipt = self.project(budget_cm=3., margin_cm=.5)
        row = receipt['contacts'][0]
        self.assertGreater(row['originalTargetCorrectionCm'], 3.)
        self.assertLessEqual(row['projectedTargetCorrectionCm'], 2.5+1e-5)
        self.assertLess(row['sourcePositionErrorCm'], 1e-5)
        self.assertLess(row['sourceOrientationErrorDegrees'], 1e-5)
        self.assertLessEqual(np.linalg.norm(root-self.root), .03)
        self.assertFalse(receipt['attempts'][0]['rootAllowed'])

    def test_impossible_target_rejects_with_failed_evidence(self):
        self.target[0]['position'] = self.target[0]['position']+np.array([1000., 0., 0.])
        old = self.local.copy()
        with self.assertRaises(TargetProjectionError) as failure:
            self.project(maximum_root_correction_meters=0.)
        self.assertFalse(failure.exception.receipt['passed'])
        self.assertFalse(failure.exception.receipt['deliverable'])
        np.testing.assert_array_equal(self.local, old)

    def test_invalid_budget_and_foreign_observer_are_rejected(self):
        for kwargs in (dict(budget_cm=10.1), dict(margin_cm=10.), dict(joint_limit_degrees=41.),
                       dict(pole_limit_degrees=91.), dict(maximum_root_correction_meters=.16)):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.project(**kwargs)
        changed = copy.deepcopy(self.rig); changed['offsets'][1][1] += .01
        with self.assertRaisesRegex(ValueError, 'different rig'):
            project_target_pose(self.local, self.root, changed, self.source, self.root,
                                self.skeleton, self.target, observer=self.observer)

    def test_source_and_target_contact_sets_must_match(self):
        self.target[0]['side'] = 'right'
        with self.assertRaisesRegex(ValueError, 'same support'):
            self.project()


class NativeMantleTargetProjectionTests(unittest.TestCase):
    def test_real_source_r5_peak_and_neighbor_observer_match_actual_conversion(self):
        run = Path(__file__).resolve().parents[2]/'workspace/runs/mantle-generalization-20261013-v1/feedback-existing-key-v1'
        if not (run/'default-sparse/round-5/authoring/receipts.json').is_file():
            self.skipTest('Optional real R5 source evidence is not present')
        native, skeleton, _ = reference_input('Mantle_High')
        native_world = world_pose(native, skeleton)
        for style in ('default-sparse', 'cautious-sparse'):
            receipts = read_json(run/style/'round-5/authoring/receipts.json')
            rig = read_json(receipts['rig:soma']['outputs']['rig.json']['path'])
            motion = load_motion(receipts['motion:Mantle_High']['outputs']['motion.npz']['path'], rig)
            reference = resample(load_motion(receipts['reference:Mantle_High:native-mantle']['outputs']['motion.npz']['path'], rig),
                                 len(motion['root_positions']), rig)
            observer = TargetPoseObserver(rig, skeleton)
            source_rot, source_pos = fk(motion['local_rot_mats'], motion['root_positions'], rig)
            for frame in (52, 54):
                with self.subTest(style=style, frame=frame):
                    hand = rig['jointNames'].index('LeftHand')
                    source = [dict(side='left', kind='hand', position=source_pos[frame, hand], rotation=source_rot[frame, hand])]
                    target = [dict(side='left', position=native_world['hand_l'][1][frame]+[0., 0., 1.],
                                   rotation=native_world['hand_l'][0][frame])]
                    self.assertTrue(observer.compare_convert(motion['local_rot_mats'][frame], motion['root_positions'][frame])['passed'])
                    result, root, receipt = project_target_pose(motion['local_rot_mats'][frame], motion['root_positions'][frame],
                        rig, source, reference['root_positions'][frame], skeleton, target, observer=observer)
                    self.assertTrue(receipt['passed'])
                    self.assertGreater(receipt['contacts'][0]['originalTargetCorrectionCm'], 10.)
                    self.assertLessEqual(receipt['contacts'][0]['projectedTargetCorrectionCm'], 9.5+1e-5)
                    self.assertEqual(receipt['rootCorrectionCm'], 0.)
                    self.assertTrue(observer.compare_convert(result, root)['passed'])


if __name__ == '__main__':
    unittest.main()

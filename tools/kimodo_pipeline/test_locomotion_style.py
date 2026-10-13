"""Posture edits must preserve gait support and reject unrelated approval."""
import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .locomotion_style import POLICY, author_posture, audit_posture, measure_posture, validate_delivery
from .motion import complete, fk, save_motion
from .runner import file_hash, write_json


def fixture():
    names = ['Hips', 'Spine1', 'Spine2', 'Chest', 'Neck1', 'Neck2', 'Head']
    parents = [-1, 0, 1, 2, 3, 4, 5]
    offsets = [[0., 0., 0.]] + [[0., .1, 0.]] * 6
    for side, sign in (('Left', 1), ('Right', -1)):
        start = len(names)
        names += [side + joint for joint in ('Shoulder', 'Arm', 'ForeArm', 'Hand')]
        parents += [3, start, start + 1, start + 2]
        offsets += [[sign * .15, 0., 0.], [sign * .08, 0., 0.], [0., -.3, 0.], [0., -.25, 0.]]
        start = len(names)
        names += [side + joint for joint in ('Leg', 'Shin', 'Foot')]
        parents += [0, start, start + 1]
        offsets += [[sign * .12, 0., 0.], [0., -.4, 0.], [0., -.4, 0.]]
    rig = dict(jointNames=names, parents=parents, offsets=offsets)
    local = np.broadcast_to(np.eye(3), (31, len(names), 3, 3)).copy()
    root = np.zeros((31, 3)); root[:, 1] = 1.; root[:, 2] = np.linspace(0., 1., 31)
    return complete(local, root, rig, 30), rig


PROFILE = dict(policy=POLICY, bodyMotionGain=.15, armMotionGain=.3, referenceFraction=.5,
               limits=dict(torsoForwardDegrees=8., neckForwardDegrees=8., headDownDegrees=8., armAbductionDegrees=12.))


class LocomotionStyleTests(unittest.TestCase):
    def test_edit_straightens_head_and_back_and_preserves_root_and_foot_fk(self):
        idle, rig = fixture()
        local = idle['local_rot_mats'].copy()
        names = {name: i for i, name in enumerate(rig['jointNames'])}
        local[:, names['Spine1']] = Rotation.from_euler('x', 30, degrees=True).as_matrix()
        local[:, names['Neck1']] = Rotation.from_euler('x', 20, degrees=True).as_matrix()
        for side, sign in (('Left', 1), ('Right', -1)):
            # Deliberate lateral arm spread, plus a changing sagittal swing.
            local[:, names[side + 'Arm']] = Rotation.from_euler('z', sign * 35, degrees=True).as_matrix()
            local[:, names[side + 'ForeArm']] = Rotation.from_euler('x', np.linspace(-15., 15., 31)[:, None], degrees=True).as_matrix()
        bad = complete(local, idle['root_positions'], rig, 30)
        self.assertFalse(audit_posture(bad, idle, rig, PROFILE)['passed'])
        edited, receipt = author_posture(bad, idle, rig, PROFILE)
        self.assertTrue(audit_posture(edited, idle, rig, PROFILE)['passed'])
        self.assertTrue(receipt['preservedRootAndLowerBody'])
        np.testing.assert_array_equal(edited['root_positions'], bad['root_positions'])
        _, before = fk(bad['local_rot_mats'], bad['root_positions'], rig)
        _, after = fk(edited['local_rot_mats'], edited['root_positions'], rig)
        lower = [i for i, name in enumerate(rig['jointNames']) if name.endswith(('Leg', 'Shin', 'Foot'))]
        np.testing.assert_array_equal(before[:, lower], after[:, lower])
        self.assertGreater(np.max(np.abs(np.diff(edited['local_rot_mats'][:, names['LeftForeArm']], axis=0))), .0001)

    def test_arbitrary_heading_and_world_translation_do_not_change_posture(self):
        idle, rig = fixture()
        source = copy.deepcopy(idle)
        yaw = Rotation.from_euler('y', 1.17).as_matrix()
        source['local_rot_mats'][:, 0] = yaw @ source['local_rot_mats'][:, 0]
        source['root_positions'] = source['root_positions'] @ yaw.T + np.array([81., 0., -24.])
        first, _ = author_posture(idle, idle, rig, PROFILE)
        second, _ = author_posture(source, idle, rig, PROFILE)
        for key, metrics in measure_posture(first, rig).items():
            for statistic, value in metrics.items():
                self.assertAlmostEqual(value, measure_posture(second, rig)[key][statistic], places=9)

    def test_bad_gains_and_incomplete_rig_are_rejected(self):
        source, rig = fixture()
        for value in (0., 1., float('nan'), True):
            profile = dict(PROFILE, bodyMotionGain=value)
            with self.assertRaises(ValueError):
                author_posture(source, source, rig, profile)
        rig['jointNames'][-1] = 'Head'
        # Missing a required upper joint also rejects rather than silently skipping it.
        rig['jointNames'][rig['jointNames'].index('LeftHand')] = 'Other'
        with self.assertRaisesRegex(ValueError, 'complete SOMA'):
            author_posture(source, source, rig, PROFILE)

    def test_delivery_rejects_missing_changed_failed_and_other_motion_evidence(self):
        motion, rig = fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload, gate = root / 'motion.npz', root / 'posture.json'
            save_motion(payload, motion, rig)
            job = dict(contract=dict(targetLocomotionPosturePolicy=POLICY), motionPath=str(payload), motionSha256=file_hash(payload))
            with self.assertRaisesRegex(ValueError, 'Missing'):
                validate_delivery(job)
            report = audit_posture(motion, motion, rig, PROFILE)
            write_json(gate, report)
            job['locomotionPostureReceipt'] = dict(file=str(gate), sha256=file_hash(gate),
                sourceMotionFile=str(payload), sourceMotionSha256=file_hash(payload), deliveredMotionSha256=file_hash(payload))
            validate_delivery(job)
            report['passed'] = False; write_json(gate, report)
            with self.assertRaisesRegex(ValueError, 'Changed'):
                validate_delivery(job)
            job['locomotionPostureReceipt']['sha256'] = file_hash(gate)
            with self.assertRaisesRegex(ValueError, 'failed'):
                validate_delivery(job)
            report.update(passed=True, motionSha256='0' * 64); write_json(gate, report)
            job['locomotionPostureReceipt']['sha256'] = file_hash(gate)
            with self.assertRaisesRegex(ValueError, 'different motion'):
                validate_delivery(job)

    def test_scope_excludes_air_actions_and_keeps_existing_contact_policies(self):
        from .graph import build_graph, load_rules, read_json, topological_order
        from .paths import contract_path, PIPELINE_ROOT
        from .locomotion_style import targets
        contracts = read_json(contract_path())
        rules = load_rules(PIPELINE_ROOT / 'tools/kimodo_pipeline/upright_feminine_rules.json')
        graph = build_graph(contracts, rules)
        order = topological_order(graph['nodes'], targets(contracts))
        semantics = {graph['nodes'][key]['params']['contract']['semanticName'] for key in targets(contracts)}
        self.assertTrue({'Walk_Forward', 'Run_Forward', 'Sprint', 'Lean_Forward', 'Stop_Left'} <= semantics)
        self.assertFalse({'Roll', 'Land_Light', 'Mantle_High', 'Air_Lean', 'Jump_Loop'} & semantics)
        self.assertFalse(any(graph['nodes'][key]['op'] == 'external' for key in order))
        generator = graph['nodes']['candidate:Run_Forward']
        self.assertEqual('posture:candidate:Run_Forward', graph['nodes']['repair:Run_Forward:generated']['params']['source'])
        self.assertEqual('motion:Idle', generator['params']['postureIdleReference'])
        self.assertEqual(.15, generator['params']['maximumAnchorCorrectionMeters'])
        self.assertEqual('als-forward-v1', graph['nodes']['asset:refactored:A_Als_Run_Forward']['params']['contract']['targetGaitPolicy'])

    def test_retained_jump_keeps_its_non_delivery_dependency_receipts(self):
        from .derived import retain_base_deliveries
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pack = root / 'assets/generated/als_nonprop_previous/resource-pack.json'
            pack.parent.mkdir(parents=True)
            author = root / 'authoring/bundle.json'; author.parent.mkdir()
            payload = root / 'retained.npz'; payload.write_bytes(b'preserved payload')
            outputs = {'motion.npz': dict(path=str(payload), sha256=file_hash(payload))}
            delivery = dict(node='asset:refactored:jump', identity='retained-id', outputs=outputs)
            rig = dict(outputs={'rig.json': dict(sha256='same-rig')})
            dependency = dict(node='dependency:Jump_Walk_Left', op='jump_dependency', identity='executed-dependency')
            incoming = {'rig:soma': rig, 'asset:refactored:jump': delivery,
                        'dependency:Jump_Walk_Left': dependency}
            write_json(author, dict(assets=[delivery]))
            write_json(author.parent / 'receipts.json', incoming)
            retarget = root / 'retarget.json'; write_json(retarget, dict(authoringBundle=str(author)))
            write_json(pack, dict(files=[], nativeRetargetBundle=str(retarget)))
            assets, receipts = retain_base_deliveries(pack, [], {'rig:soma': rig})
            self.assertEqual(delivery['identity'], assets[0]['identity'])
            self.assertEqual(file_hash(author.parent / 'receipts.json'), assets[0]['sourceAuthoringReceipts']['sha256'])
            self.assertEqual(dependency, receipts['dependency:Jump_Walk_Left'])

    def test_fixed_upper_arms_leave_visible_forearm_motion_and_preserve_crouched_back(self):
        from .locomotion_forearms import POLICY as FOREARMS, arm_motion
        idle, rig = fixture()
        names = {name: i for i, name in enumerate(rig['jointNames'])}
        local = idle['local_rot_mats'].copy()
        local[:, names['Spine1']] = Rotation.from_euler('x', 40, degrees=True).as_matrix()
        local[:, names['LeftArm']] = Rotation.from_euler('x', np.linspace(-25, 25, 31)[:, None], degrees=True).as_matrix()
        source = complete(local, idle['root_positions'], rig, 30)
        profile = dict(policy=FOREARMS, bodyMotionGain=.15, forearmMotionGain=.85,
            forearmSwingDegrees=22., elbowRestDegrees=45., referenceFraction=.5,
            stanceMode='crouching', forearmSwing=True,
            limits=dict(PROFILE['limits'], upperArmMotionDegrees=.05, minimumHandTravelMeters=.08))
        result, receipt = author_posture(source, idle, rig, profile)
        self.assertTrue(audit_posture(result, idle, rig, profile)['passed'])
        spine = [names[name] for name in ('Spine1', 'Spine2', 'Chest')]
        np.testing.assert_array_equal(result['local_rot_mats'][:, spine], source['local_rot_mats'][:, spine])
        self.assertTrue(receipt['preservedCrouchSpine'])
        for side, metrics in arm_motion(result, rig).items():
            self.assertLess(metrics['upperArmMotionDegrees'], 1e-10)
            self.assertGreater(metrics['handTravelMeters'], .15)
        self.assertGreater(measure_posture(result, rig)['torsoForwardDegrees']['median'], 35.)
        _, before = fk(source['local_rot_mats'], source['root_positions'], rig)
        _, after = fk(result['local_rot_mats'], result['root_positions'], rig)
        legs = [i for i, name in enumerate(rig['jointNames']) if name.endswith(('Leg', 'Shin', 'Foot'))]
        np.testing.assert_array_equal(before[:, legs], after[:, legs])

    def test_forearm_gate_rejects_frozen_hands_and_swinging_upper_arms(self):
        from .locomotion_forearms import POLICY as FOREARMS
        idle, rig = fixture()
        profile = dict(policy=FOREARMS, bodyMotionGain=.15, forearmMotionGain=.85,
            forearmSwingDegrees=22., elbowRestDegrees=45., referenceFraction=.5,
            stanceMode='standing', forearmSwing=True,
            limits=dict(PROFILE['limits'], upperArmMotionDegrees=.05, minimumHandTravelMeters=.08))
        frozen = audit_posture(idle, idle, rig, profile)
        self.assertFalse(frozen['passed'])
        self.assertEqual(2, sum(not c['passed'] for c in frozen['checks']))
        styled, _ = author_posture(idle, idle, rig, profile)
        local = styled['local_rot_mats'].copy()
        arm = rig['jointNames'].index('LeftArm')
        local[:, arm] = Rotation.from_euler('x', np.linspace(-10, 10, 31)[:, None], degrees=True).as_matrix()
        bad = complete(local, styled['root_positions'], rig, 30)
        gate = audit_posture(bad, idle, rig, profile)
        self.assertFalse(next(c for c in gate['checks'] if c['metric'] == 'LeftUpperArmMotionDegrees')['passed'])

    def test_idle_stabilization_preserves_breathing_and_forearm_changes(self):
        from .locomotion_forearms import POLICY as FOREARMS,author_idle_arms,arm_motion
        idle,rig=fixture();names={n:i for i,n in enumerate(rig['jointNames'])}
        local=idle['local_rot_mats'].copy()
        for name,amplitude in [('Spine2',2.),('LeftArm',12.),('LeftForeArm',6.)]:
            local[:,names[name]]=Rotation.from_euler('x',np.linspace(-amplitude,amplitude,31)[:,None],degrees=True).as_matrix()
        source=complete(local,idle['root_positions'],rig,30)
        profile=dict(policy=FOREARMS,bodyMotionGain=.15,forearmMotionGain=.85,
            forearmSwingDegrees=22.,elbowRestDegrees=45.,referenceFraction=.5,stanceMode='standing',forearmSwing=False,
            limits=dict(PROFILE['limits'],upperArmMotionDegrees=.05,minimumHandTravelMeters=.08))
        result,report=author_idle_arms(source,rig,profile)
        self.assertTrue(report['preservedIdleBodyAndForearms'])
        for name in ('Spine1','Spine2','Chest','Neck1','Head','LeftForeArm'):
            np.testing.assert_array_equal(result['local_rot_mats'][:,names[name]],source['local_rot_mats'][:,names[name]])
        self.assertLess(arm_motion(result,rig)['Left']['upperArmMotionDegrees'],1e-10)


if __name__ == '__main__':
    unittest.main()

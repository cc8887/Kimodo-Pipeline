"""Geometric regressions for missing target twist and branch discontinuities."""
import copy
import unittest
import hashlib
import tempfile
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from .twist_reconstruction import TwistReconstruction, reconstruct, reconstruct_relative, SEGMENTS, ADDITIVE_POLICY
from .runner import write_json, file_hash
from unittest.mock import patch


def fixture(count=3, axis=(0., 1., 0.)):
    names = ['root']; parents = [-1]
    reference = [dict(position=[0., 0., 0.], rotation=[0., 0., 0., 1.], scale=[1., 1., 1.])]
    for side in ('l', 'r'):
        for a, b, c in [('upperarm', 'lowerarm', 'hand'), ('thigh', 'calf', 'foot')]:
            start = len(names); sign = 1 if side == 'l' else -1
            for name, parent, distance in [(a + '_' + side, 0, 0), (b + '_' + side, start, 10),
                                          (c + '_' + side, start + 1, 10),
                                          (a + '_twist_01_' + side, start, 2),
                                          (b + '_twist_01_' + side, start + 1, 8)]:
                names.append(name); parents.append(parent)
                reference.append(dict(position=(np.asarray(axis) * distance * sign).tolist(),
                                      rotation=[0., 0., 0., 1.], scale=[1., 1., 1.]))
    skeleton = dict(rawBoneNames=names, rawParents=parents, referencePose=reference)
    raw = dict(sampledKeyCount=count, playLength=(count - 1) / 30, frameRateNumerator=30,
               frameRateDenominator=1, tracks=[dict(bone=n, positions=[r['position']],
               rotations=[r['rotation']], scales=[r['scale']]) for n, r in zip(names, reference)])
    return skeleton, raw


class TwistReconstructionTests(unittest.TestCase):
    def test_overlay_slot_assembly_repairs_stale_leaf_twist_and_keeps_body_tracks(self):
        from .publish import assemble_overlay_lower_body
        from .twist_reconstruction import POLICY
        skeleton,table=fixture(4,axis=(1.,0.,0.))
        _,stand=fixture(1,axis=(1.,0.,0.));crouch=copy.deepcopy(stand)
        next(t for t in crouch['tracks'] if t['bone']=='foot_l')['rotations']=[Rotation.from_euler('x',60.,degrees=True).as_quat().tolist()]
        jobs={source:dict(edition='v4',contract=dict(semanticName=semantic))
              for source,semantic in [('table','Default_Poses'),('stand','Stand_Pose'),('crouch','Crouch_Pose')]}
        jobs['table']['twistReconstructionPolicy']=POLICY
        rewritten=dict(table=copy.deepcopy(table),stand=stand,crouch=crouch)
        def factor(raw,skeleton,job):
            job['_rootOwnershipReport']={};return raw
        with patch('tools.kimodo_pipeline.publish.target_root_and_ik',side_effect=factor):
            assemble_overlay_lower_body(jobs,rewritten,{'v4':skeleton})
        self.assertTrue(TwistReconstruction(skeleton).assess(rewritten['table'])['passed'])
        self.assertEqual(jobs['table']['_targetTwistReport']['repairStage'],'after-overlay-slot-assembly')
        actual={t['bone']:t for t in rewritten['table']['tracks']}
        for track in table['tracks']:
            if 'twist' not in track['bone'] and not track['bone'].startswith(('thigh_','calf_','foot_','ball_')):
                self.assertEqual(actual[track['bone']],track)
        self.assertEqual(actual['foot_l']['rotations'],[stand['tracks'][8]['rotations'][0]]*2+[next(t for t in crouch['tracks'] if t['bone']=='foot_l')['rotations'][0]]*2)

    def test_relative_identity_preserves_legacy_baseline_without_claiming_absolute_conformance(self):
        skeleton,base=fixture(1,axis=(1.,0.,0.))
        for track in base['tracks']:
            if 'twist' in track['bone']:
                track['rotations']=[Rotation.from_euler('x',23.,degrees=True).as_quat().tolist()]
        raw=copy.deepcopy(base);raw.update(sampledKeyCount=3,playLength=2/30)
        fixed,report=reconstruct_relative(raw,skeleton,base,['thigh_l','thigh_r'])
        self.assertTrue(report['passed']);self.assertEqual(ADDITIVE_POLICY,report['policy'])
        self.assertFalse(report['baselineAbsoluteConformancePassed'])
        self.assertFalse(TwistReconstruction(skeleton).assess(fixed)['passed'])
        for track in fixed['tracks']:
            if 'twist' in track['bone']:
                actual=Rotation.from_quat(track['rotations']).as_matrix()
                np.testing.assert_allclose(actual,np.broadcast_to(Rotation.from_euler('x',23.,degrees=True).as_matrix(),actual.shape),atol=1e-12)

    def test_relative_owned_leg_delta_does_not_rebuild_unowned_arm_twist(self):
        skeleton,base=fixture(1,axis=(1.,0.,0.))
        for track in base['tracks']:
            if 'twist' in track['bone']:track['rotations']=[Rotation.from_euler('x',17.,degrees=True).as_quat().tolist()]
        raw=copy.deepcopy(base);raw.update(sampledKeyCount=3,playLength=2/30)
        next(t for t in raw['tracks'] if t['bone']=='foot_l')['rotations']=Rotation.from_euler('x',np.array([0.,40.,80.])[:,None],degrees=True).as_quat().tolist()
        # Simulate the faulty global reconstruction that erased upper-body base twist.
        absolute,_=reconstruct(raw,skeleton)
        corrected,report=reconstruct_relative(absolute,skeleton,base,['thigh_l','thigh_r'])
        self.assertTrue(report['passed'])
        by_name={t['bone']:t for t in corrected['tracks']}
        np.testing.assert_allclose(Rotation.from_quat(by_name['calf_twist_01_l']['rotations']).as_matrix(),
            Rotation.from_euler('x',np.array([17.,37.,57.])[:,None],degrees=True).as_matrix(),atol=1e-12)
        np.testing.assert_allclose(Rotation.from_quat(by_name['upperarm_twist_01_l']['rotations']).as_matrix(),
            np.broadcast_to(Rotation.from_euler('x',17.,degrees=True).as_matrix(),(3,3,3)),atol=1e-12)
        for original in absolute['tracks']:
            if 'twist' not in original['bone']:self.assertEqual(original,by_name[original['bone']])
        bad=copy.deepcopy(corrected)
        next(t for t in bad['tracks'] if t['bone']=='upperarm_twist_01_l')['rotations']=[[0,0,0,1]]
        self.assertFalse(TwistReconstruction(skeleton).assess_relative(bad,base,['thigh_l','thigh_r'])['passed'])

    def test_relative_half_twist_branch_is_anchored_to_the_actual_base(self):
        skeleton,base=fixture(1,axis=(1.,0.,0.))
        next(t for t in base['tracks'] if t['bone']=='foot_l')['rotations']=Rotation.from_euler('x',170,degrees=True).as_quat()[None].tolist()
        raw=copy.deepcopy(base);raw.update(sampledKeyCount=3,playLength=2/30)
        next(t for t in raw['tracks'] if t['bone']=='foot_l')['rotations']=Rotation.from_euler('x',np.array([170.,190.,210.])[:,None],degrees=True).as_quat().tolist()
        corrected,report=reconstruct_relative(raw,skeleton,base,['thigh_l'])
        twist=next(t for t in corrected['tracks'] if t['bone']=='calf_twist_01_l')
        np.testing.assert_allclose(Rotation.from_quat(twist['rotations']).as_matrix(),Rotation.from_euler('x',np.array([0.,10.,20.])[:,None],degrees=True).as_matrix(),atol=1e-12)
        self.assertTrue(report['passed'])

    def test_actual_segment_axis_mirror_and_proximal_distal_roles(self):
        skeleton, raw = fixture()
        tracks = {t['bone']: t for t in raw['tracks']}
        for side, sign in [('l', 1), ('r', -1)]:
            tracks['upperarm_' + side]['rotations'] = Rotation.from_euler('y', sign * 90, degrees=True).as_quat()[None].tolist()
            tracks['hand_' + side]['rotations'] = Rotation.from_euler('y', sign * 60, degrees=True).as_quat()[None].tolist()
        source = copy.deepcopy(raw); result, report = reconstruct(raw, skeleton)
        self.assertEqual(8, report['physicalTwistBones']); self.assertTrue(report['passed']); self.assertEqual(source, raw)
        output = {t['bone']: t for t in result['tracks']}
        for side, sign in [('l', 1), ('r', -1)]:
            for bone, angle in [('upperarm', -45), ('lowerarm', 30)]:
                actual = Rotation.from_quat(output[bone + '_twist_01_' + side]['rotations']).as_matrix()
                expected = Rotation.from_euler('y', sign * angle, degrees=True).as_matrix()
                np.testing.assert_allclose(actual, np.broadcast_to(expected, actual.shape), atol=1e-12)
        # FK endpoints and fingers keep precisely the same local inputs.
        for track in raw['tracks']:
            if 'twist' not in track['bone']: self.assertEqual(track, output[track['bone']])
            else:
                self.assertEqual(track['positions'], output[track['bone']]['positions'])
                self.assertEqual(track['scales'], output[track['bone']]['scales'])

    def test_swing_is_not_axial_roll_and_nonidentity_bind_axes_are_preserved(self):
        skeleton, raw = fixture(axis=(1., 0., 0.))
        index = skeleton['rawBoneNames'].index('upperarm_l')
        rest = Rotation.from_euler('z', 40, degrees=True)
        skeleton['referencePose'][index]['rotation'] = rest.as_quat().tolist()
        track = next(t for t in raw['tracks'] if t['bone'] == 'upperarm_l')
        track['rotations'] = (rest * Rotation.from_euler('y', np.array([20., 60., 100.])[:, None], degrees=True)).as_quat().tolist()
        result, report = reconstruct(raw, skeleton)
        self.assertTrue(report['passed'])
        twist = next(t for t in result['tracks'] if t['bone'] == 'upperarm_twist_01_l')
        np.testing.assert_allclose(twist['rotations'], [[0., 0., 0., 1.]] * 3, atol=1e-12)
        # A bind pose reconstructs to the original twist reference, not identity.
        twist_index = skeleton['rawBoneNames'].index('lowerarm_twist_01_l')
        skeleton['referencePose'][twist_index]['rotation'] = rest.as_quat().tolist()
        result, _ = reconstruct(raw, skeleton)
        twist = next(t for t in result['tracks'] if t['bone'] == 'lowerarm_twist_01_l')
        np.testing.assert_allclose(twist['rotations'], np.tile(rest.as_quat(), (3, 1)), atol=1e-12)

    def test_half_twist_unwraps_across_180_and_ignores_quaternion_sign(self):
        skeleton, raw = fixture(axis=(1., 0., 0.))
        hand = next(t for t in raw['tracks'] if t['bone'] == 'hand_l')
        q = Rotation.from_euler('x', np.array([170., 190., 210.])[:, None], degrees=True).as_quat(); q[1] *= -1
        hand['rotations'] = q.tolist()
        result, report = reconstruct(raw, skeleton)
        self.assertTrue(report['passed'])
        twist = next(t for t in result['tracks'] if t['bone'] == 'lowerarm_twist_01_l')
        expected = Rotation.from_euler('x', np.array([85., 95., 105.])[:, None], degrees=True).as_matrix()
        np.testing.assert_allclose(Rotation.from_quat(twist['rotations']).as_matrix(), expected, atol=1e-12)
        row = next(r for r in report['bones'] if r['bone'] == twist['bone'])
        self.assertAlmostEqual(10., row['maximumStepDegrees'], places=10)

    def test_constant_twist_cannot_pass_for_a_rotating_wrist(self):
        skeleton, raw = fixture(axis=(1., 0., 0.))
        next(t for t in raw['tracks'] if t['bone'] == 'hand_l')['rotations'] = Rotation.from_euler('x', np.array([0., 60., 120.])[:, None], degrees=True).as_quat().tolist()
        self.assertFalse(TwistReconstruction(skeleton).assess(raw)['passed'])
        self.assertTrue(reconstruct(raw, skeleton)[1]['passed'])

    def test_missing_or_nonleaf_twist_and_undefined_swing_fail_closed(self):
        skeleton, raw = fixture(axis=(1., 0., 0.))
        bad = copy.deepcopy(skeleton); bad['rawParents'][-1] = len(bad['rawParents']) - 2
        with self.assertRaisesRegex(ValueError, 'leaf sibling'): TwistReconstruction(bad)
        bad = copy.deepcopy(skeleton); bad['rawBoneNames'][-1] = 'unexpected'
        with self.assertRaisesRegex(ValueError, 'Missing ALS twist'): TwistReconstruction(bad)
        next(t for t in raw['tracks'] if t['bone'] == 'hand_l')['rotations'] = Rotation.from_euler('y', 180., degrees=True).as_quat()[None].tolist()
        with self.assertRaisesRegex(ValueError, 'Undefined 180-degree swing'): reconstruct(raw, skeleton)

    def test_pack_preflight_recomputes_final_twist_even_if_a_flat_track_has_fresh_hashes(self):
        from .publish import ROOT, validate_twist_delivery
        from .root_ownership import semantic_hash
        from .twist_reconstruction import POLICY
        skeleton, raw = fixture(axis=(1., 0., 0.)); source = '/test/mantle'
        next(t for t in raw['tracks'] if t['bone'] == 'hand_l')['rotations'] = Rotation.from_euler('x',
            np.array([0., 60., 120.])[:, None], degrees=True).as_quat().tolist()
        raw, report = reconstruct(raw, skeleton)
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as directory:
            folder = Path(directory); config = folder/'config'; config.mkdir()
            write_json(config/'v4_recovery_movement_source_inputs.json', dict(skeletons=[skeleton]))
            write_json(config/'refactored_animation_sources.json', dict(skeletons={'/ALS/ALS/Character/SK_Als.SK_Als': skeleton}))
            job_path = folder/'jobs.json'
            write_json(job_path, dict(jobs=[dict(source=source, edition='v4', twistReconstructionPolicy=POLICY)]))
            raw_path = folder/'root-ownership'/hashlib.sha256(source.encode()).hexdigest()/'runtime.raw.json'
            raw_path.parent.mkdir(parents=True); write_json(raw_path, raw)
            report_path = folder/'target-twist.json'
            write_json(report_path, dict(**report, source=source, finalRawSemanticSha256=semantic_hash(raw)))
            manifest = dict(nativeRetargetBundle=str(job_path), targetTwistChecks=[dict(source=source,
                policy=POLICY, passed=True, file=report_path.name)], files=[])
            def hashes():
                manifest['files'] = [dict(path='res://'+p.relative_to(ROOT).as_posix(), sha256=file_hash(p))
                                     for p in (report_path, raw_path)]
            hashes()
            with patch('tools.kimodo_pipeline.publish.CONFIG', config):
                self.assertTrue(validate_twist_delivery(folder/'resource-pack.json', manifest)['passed'])
                # Simulate re-stamping a deficient resource as passing.
                next(t for t in raw['tracks'] if t['bone'] == 'lowerarm_twist_01_l')['rotations'] = [[0., 0., 0., 1.]]
                write_json(raw_path, raw); hashes()
                with self.assertRaisesRegex(ValueError, 'conformance failed'):
                    validate_twist_delivery(folder/'resource-pack.json', manifest)
                manifest['targetTwistChecks'] = []
                with self.assertRaisesRegex(ValueError, 'per-asset twist gates'):
                    validate_twist_delivery(folder/'resource-pack.json', manifest)

    def test_relative_pack_preflight_binds_actual_base_and_rejects_restamped_nonowned_twist(self):
        from .publish import ROOT,validate_twist_delivery,assess_target_twist
        from .root_ownership import semantic_hash
        from .twist_reconstruction import POLICY
        skeleton,base=fixture(1,axis=(1.,0.,0.));source='/test/correction';base_source='/test/legacy-base'
        base['source']=base_source
        for track in base['tracks']:
            if 'twist' in track['bone']:track['rotations']=Rotation.from_euler('x',23,degrees=True).as_quat()[None].tolist()
        raw=copy.deepcopy(base);raw.update(source=source,sampledKeyCount=3,playLength=2/30)
        raw,_=reconstruct_relative(raw,skeleton,base,['thigh_l','thigh_r'])
        spec=dict(runtimeBaseSource=base_source,runtimeBaseFrame=0,correctionBoneRoots=['thigh_l','thigh_r'])
        job=dict(source=source,edition='v4',twistReconstructionPolicy=POLICY,contract=dict(targetAdditiveRebase=spec))
        base_job=dict(source=base_source,edition='v4',contract=dict(originalEvaluation=dict(interpolation='Linear')))
        report=assess_target_twist(raw,skeleton,job,base,base_job)
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as directory:
            folder=Path(directory);config=folder/'config';config.mkdir()
            write_json(config/'v4_recovery_movement_source_inputs.json',dict(skeletons=[skeleton]))
            write_json(config/'refactored_animation_sources.json',dict(skeletons={'/ALS/ALS/Character/SK_Als.SK_Als':skeleton}))
            job_path=folder/'jobs.json';write_json(job_path,dict(jobs=[job,base_job]))
            raw_path=folder/'root-ownership'/hashlib.sha256(source.encode()).hexdigest()/'runtime.raw.json'
            base_path=folder/'root-ownership'/hashlib.sha256(base_source.encode()).hexdigest()/'runtime.raw.json'
            raw_path.parent.mkdir(parents=True);base_path.parent.mkdir(parents=True)
            write_json(raw_path,raw);write_json(base_path,base)
            report_path=folder/'target-twist.json'
            write_json(report_path,dict(**report,source=source,finalRawSemanticSha256=semantic_hash(raw)))
            manifest=dict(nativeRetargetBundle=str(job_path),targetTwistChecks=[dict(source=source,
                policy=ADDITIVE_POLICY,passed=True,file=report_path.name)],files=[])
            def hashes():
                manifest['files']=[dict(path='res://'+p.relative_to(ROOT).as_posix(),sha256=file_hash(p))
                                   for p in (report_path,raw_path,base_path)]
            hashes()
            with patch('tools.kimodo_pipeline.publish.CONFIG',config):
                verdict=validate_twist_delivery(folder/'resource-pack.json',manifest)
                self.assertTrue(verdict['passed']);self.assertEqual([ADDITIVE_POLICY],verdict['assetPolicies'])
                next(t for t in raw['tracks'] if t['bone']=='upperarm_twist_01_l')['rotations']=[[0,0,0,1]]
                write_json(raw_path,raw);hashes()
                with self.assertRaisesRegex(ValueError,'conformance failed'):
                    validate_twist_delivery(folder/'resource-pack.json',manifest)
                manifest['files']=[entry for entry in manifest['files'] if not entry['path'].endswith(base_path.parent.name+'/runtime.raw.json')]
                with self.assertRaisesRegex(ValueError,'Missing or changed twist payload'):
                    validate_twist_delivery(folder/'resource-pack.json',manifest)


if __name__ == '__main__': unittest.main()

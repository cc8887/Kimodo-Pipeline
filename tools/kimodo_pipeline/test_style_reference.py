"""Optional style anchors must preserve native locomotion and remain references."""
import copy
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from .gait import _array_hash, foot_landmarks
from .motion import complete, fk
from .style_reference import compose_upper_body_reference


def fixture():
    names = ['Hips','Spine1','Spine2','Chest','Neck1','Neck2','Head']
    parents = [-1,0,1,2,3,4,5]
    offsets = [[0.,0.,0.]]+[[0.,.1,0.]]*6
    for side, sign in [('Left',1.),('Right',-1.)]:
        start=len(names)
        names += [side+name for name in ('Shoulder','Arm','ForeArm','Hand','HandIndexEnd')]
        parents += [3,start,start+1,start+2,start+3]
        offsets += [[sign*.1,.04,0.],[sign*.1,0.,0.],[sign*.3,0.,0.],[sign*.25,0.,0.],[sign*.07,0.,0.]]
        start=len(names)
        names += [side+name for name in ('Leg','Shin','Foot','ToeBase','ToeEnd')]
        parents += [0,start,start+1,start+2,start+3]
        offsets += [[sign*.1,0.,0.],[0.,-.4,0.],[0.,-.4,0.],[0.,-.08,.05],[0.,0.,.1]]
    rig=dict(jointNames=names,parents=parents,offsets=offsets)
    local=np.tile(np.eye(3),(19,len(names),1,1))
    root=np.column_stack([np.linspace(2.,3.,19),np.full(19,.9),np.linspace(3.,4.,19)])
    local[:,0]=Rotation.from_euler('yx',np.column_stack([np.linspace(-.2,.3,19),np.full(19,.2)])).as_matrix()
    local[:,names.index('LeftLeg')]=Rotation.from_euler('x',np.linspace(-.2,.2,19)[:,None]).as_matrix()
    native=complete(local,root,rig,30)
    style_local=np.tile(np.eye(3),(7,len(names),1,1))
    style_local[:,0]=Rotation.from_euler('y',.9).as_matrix()
    style_local[:,names.index('Spine1')]=Rotation.from_euler('x',np.linspace(-.1,.4,7)[:,None]).as_matrix()
    style_local[:,names.index('LeftArm')]=Rotation.from_euler('z',-.8).as_matrix()
    style_local[:,names.index('RightArm')]=Rotation.from_euler('z',.65).as_matrix()
    style_local[:,names.index('LeftForeArm')]=Rotation.from_euler('y',np.linspace(.2,1.1,7)[:,None]).as_matrix()
    style=complete(style_local,np.tile([20.,1.,-7.],(7,1)),rig,60)
    return native,style,rig


class StyleReferenceTests(unittest.TestCase):
    def test_global_style_aligns_to_heading_without_changing_root_hips_legs_or_feet(self):
        native,style,rig=fixture()
        output,receipt=compose_upper_body_reference(native,style,rig)
        preserved=[rig['jointNames'].index(name) for name in receipt['preservedJoints']]
        self.assertEqual(output['local_rot_mats'][:,preserved].tobytes(),native['local_rot_mats'][:,preserved].tobytes())
        self.assertEqual(output['root_positions'].tobytes(),native['root_positions'].tobytes())
        np.testing.assert_array_equal(foot_landmarks(output,rig)[0],foot_landmarks(native,rig)[0])
        before,_=fk(native['local_rot_mats'],native['root_positions'],rig)
        sampled,_=fk(style['local_rot_mats'],style['root_positions'],rig)
        after,_=fk(output['local_rot_mats'],output['root_positions'],rig)
        yaw=np.arctan2(before[:,0,0,2],before[:,0,2,2])-.9
        align=Rotation.from_euler('y',yaw[:,None]).as_matrix()
        for name in receipt['selectedUpperJoints']:
            joint=rig['jointNames'].index(name)
            np.testing.assert_allclose(after[:,joint],align@sampled[3,joint],atol=1e-12,rtol=0)
        self.assertEqual(receipt['maximumFootFkErrorMeters'],0.)
        self.assertFalse(receipt['deliverable'])
        self.assertFalse(receipt['productionReady'])
        self.assertFalse(receipt['referenceAcceptedAsMaster'])
        self.assertEqual(receipt['styleQualityStatus'],'not_evaluated')

    def test_different_style_or_sample_changes_upper_pose_without_fixed_idle_policy(self):
        native,style,rig=fixture()
        first,a=compose_upper_body_reference(native,style,rig,fraction=0.)
        last,b=compose_upper_body_reference(native,style,rig,fraction=1.)
        self.assertEqual(a['styleFrame'],0)
        self.assertEqual(b['styleFrame'],6)
        self.assertEqual(b['styleSampleTimeSeconds'],.1)
        self.assertNotEqual(a['outputMotionSha256'],b['outputMotionSha256'])
        arm=rig['jointNames'].index('LeftForeArm')
        self.assertGreater(np.linalg.norm(first['global_rot_mats'][0,arm]-last['global_rot_mats'][0,arm]),.5)

    def test_inputs_remain_unchanged_and_receipt_binds_both_actual_sources(self):
        native,style,rig=fixture()
        native_copy,style_copy=copy.deepcopy(native),copy.deepcopy(style)
        output,receipt=compose_upper_body_reference(native,style,rig)
        for source,unchanged in [(native,native_copy),(style,style_copy)]:
            for key,value in source.items():
                np.testing.assert_array_equal(value,unchanged[key])
        hash_of=lambda value:_array_hash(value['local_rot_mats'],value['root_positions'],value['fps'])
        self.assertEqual(receipt['nativeMotionSha256'],hash_of(native))
        self.assertEqual(receipt['styleMotionSha256'],hash_of(style))
        self.assertEqual(receipt['outputMotionSha256'],hash_of(output))
        output['root_positions'][0,0]+=1.
        np.testing.assert_array_equal(native['root_positions'],native_copy['root_positions'])

    def test_missing_or_wrong_rig_binding_is_rejected(self):
        native,style,rig=fixture()
        wrong=copy.deepcopy(style);wrong['joint_names']=list(reversed(rig['jointNames']))
        with self.assertRaisesRegex(ValueError,'joint order'):
            compose_upper_body_reference(native,wrong,rig)
        wrong=copy.deepcopy(style);wrong['rigSha256']='0'*64
        with self.assertRaisesRegex(ValueError,'rig digest'):
            compose_upper_body_reference(native,wrong,rig)
        wrong=copy.deepcopy(style);del wrong['posed_joints']
        with self.assertRaisesRegex(ValueError,'complete pose/rig'):
            compose_upper_body_reference(native,wrong,rig)
        wrong_rig=copy.deepcopy(rig);wrong_rig['offsets'][rig['jointNames'].index('LeftLeg')][0]+=.1
        with self.assertRaisesRegex(ValueError,'complete pose/rig'):
            compose_upper_body_reference(native,style,wrong_rig)

    def test_incomplete_hierarchy_and_ambiguous_heading_are_rejected(self):
        native,style,rig=fixture()
        wrong=copy.deepcopy(rig);wrong['jointNames'][2]='MissingSpine2'
        with self.assertRaisesRegex(ValueError,'SOMA chain'):
            compose_upper_body_reference(native,style,wrong)
        local=style['local_rot_mats'].copy();local[:,0]=Rotation.from_euler('x',90.,degrees=True).as_matrix()
        vertical=complete(local,style['root_positions'],rig,float(style['fps']))
        with self.assertRaisesRegex(ValueError,'heading is undefined'):
            compose_upper_body_reference(native,vertical,rig)

    def test_invalid_fraction_and_stale_fk_are_rejected(self):
        native,style,rig=fixture()
        for fraction in (True,-.1,1.1,float('nan')):
            with self.subTest(fraction=fraction):
                with self.assertRaisesRegex(ValueError,'fraction'):
                    compose_upper_body_reference(native,style,rig,fraction=fraction)
        stale=copy.deepcopy(style);stale['posed_joints'][0,0,0]+=.01
        with self.assertRaisesRegex(ValueError,'complete pose/rig'):
            compose_upper_body_reference(native,stale,rig)

    def test_actual_generated_pose_keeps_native_phase_evidence_valid(self):
        from .als_reference import native_gait_reference
        from .constraint_phase import build_native_phase_evidence, validate_native_phase_evidence
        from .graph import read_json
        from .motion import load_motion
        from .paths import project_root, reference_root
        from .runner import file_hash
        receipts=read_json(project_root()/'artifacts/kimodo_als/nonprop-regeneration-20261009-v1/combined-receipts.json')
        rig=read_json(receipts['rig:soma']['outputs']['rig.json']['path'])
        style_receipt=receipts['candidate:Idle']
        self.assertEqual(style_receipt['op'],'generate')
        entry=style_receipt['outputs']['motion.npz']
        self.assertEqual(file_hash(entry['path']),entry['sha256'])
        style=load_motion(entry['path'],rig)
        index_path=reference_root()/'assets/config/refactored_animation_sources.json'
        source=next(row['source'] for row in read_json(index_path)['assets'] if row['source'].endswith('.A_Als_Sprint'))
        native,original=native_gait_reference(index_path,source,rig,6.,[0.,1.])
        output,receipt=compose_upper_body_reference(native,style,rig)
        evidence=build_native_phase_evidence(index_path,source,output,rig)
        self.assertEqual(validate_native_phase_evidence(evidence,output,rig)['sha256'],evidence['sha256'])
        self.assertFalse(original['referenceAudit']['pass'])
        self.assertFalse(evidence['referenceAcceptedAsMaster'])
        self.assertEqual(receipt['maximumFootFkErrorMeters'],0.)
        self.assertEqual(evidence['referenceMotionSha256'],receipt['outputMotionSha256'])


if __name__=='__main__':
    unittest.main()

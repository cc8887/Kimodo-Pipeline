"""Conditioning projection preserves local style and rejects infeasible contacts."""
import copy
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

from .mantle_constraint_feedback import ProjectionError, build_feedback_constraints, project_pose, select_failure_keys
from .motion import complete, fk


def rig_fixture():
    names=['Hips','Spine1','Spine2','Chest','Head','LeftShoulder','LeftArm','LeftForeArm','LeftHand',
           'LeftFinger','RightShoulder','RightArm','RightForeArm','RightHand','RightFinger',
           'LeftLeg','LeftShin','LeftFoot','RightLeg','RightShin','RightFoot']
    parents=[-1,0,1,2,3,3,5,6,7,8,3,10,11,12,13,0,15,16,0,18,19]
    offsets=[[0,0,0],[0,.3,0],[0,.25,0],[0,.25,0],[0,.2,0],[-.1,0,0],[-.1,0,0],[-.25,0,0],[-.25,0,0],[-.1,0,0],
             [.1,0,0],[.1,0,0],[.25,0,0],[.25,0,0],[.1,0,0],[-.1,0,0],[0,-.4,0],[0,-.4,0],[.1,0,0],[0,-.4,0],[0,-.4,0]]
    return dict(jointNames=names,parents=parents,offsets=offsets)


class MantleConstraintFeedbackTests(unittest.TestCase):
    def setUp(self):
        self.rig=rig_fixture()
        self.local=np.repeat(np.eye(3)[None],len(self.rig['jointNames']),axis=0)
        self.local[4]=Rotation.from_euler('y',23,degrees=True).as_matrix()
        self.local[9]=Rotation.from_euler('y',16,degrees=True).as_matrix()
        self.local[14]=Rotation.from_euler('x',-12,degrees=True).as_matrix()

    def test_root_tube_projection_does_not_replace_free_pose(self):
        result,root,receipt=project_pose(self.local,np.array([0,.4,0]),self.rig,np.zeros(3),[])
        np.testing.assert_array_equal(result,self.local)
        self.assertAlmostEqual(np.linalg.norm(root),.135,places=7)
        self.assertAlmostEqual(receipt['rootCorrectionCm'],26.5,places=5)
        self.assertFalse(receipt['deliverable'])
        self.assertTrue(receipt['passed'])

    def test_two_support_hands_become_reachable_without_changing_free_joints_or_lengths(self):
        before=copy.deepcopy(self.rig)
        contacts=[dict(kind='hand',side=side,position=[x,.8,.65],rotation=np.eye(3))
                  for side,x in (('left',-.3),('right',.3))]
        result,root,receipt=project_pose(self.local,np.zeros(3),self.rig,np.zeros(3),contacts)
        self.assertEqual(self.rig,before)
        self.assertLessEqual(np.linalg.norm(root),.15)
        for index in (4,9,14,15,16,17,18,19,20):
            np.testing.assert_array_equal(result[index],self.local[index])
        _,points=fk(result[None],root[None],self.rig)
        for contact in contacts:
            index=self.rig['jointNames'].index(contact['side'].title()+'Hand')
            np.testing.assert_allclose(points[0,index],contact['position'],atol=1e-7)
        for index,parent in enumerate(self.rig['parents']):
            if parent>=0:
                self.assertAlmostEqual(np.linalg.norm(points[0,index]-points[0,parent]),np.linalg.norm(self.rig['offsets'][index]),places=7)
        self.assertTrue(any(row['name'] in ('Hips','Spine1','Spine2','Chest') for row in receipt['affectedJoints']))

    def test_landing_feet_use_same_fixed_length_contact_solver(self):
        contacts=[dict(kind='foot',side=side,position=[x,-.72,.08],rotation=np.eye(3))
                  for side,x in (('left',-.1),('right',.1))]
        result,root,receipt=project_pose(self.local,np.zeros(3),self.rig,np.zeros(3),contacts)
        self.assertTrue(receipt['passed'])
        for index in range(15):
            np.testing.assert_array_equal(result[index],self.local[index])
        self.assertTrue(all(item['afterPositionErrorCm']<1e-4 for item in receipt['contacts']))

    def test_impossible_contacts_raise_with_failure_evidence_and_leave_inputs_unchanged(self):
        before=self.local.copy()
        with self.assertRaises(ProjectionError) as error:
            project_pose(self.local,np.zeros(3),self.rig,np.zeros(3),[
                dict(kind='hand',side='left',position=[-10,0,0],rotation=np.eye(3))])
        self.assertFalse(error.exception.receipt['passed'])
        np.testing.assert_array_equal(self.local,before)

    def test_projection_is_deterministic(self):
        contacts=[dict(kind='hand',side='left',position=[-.3,.8,.6],rotation=np.eye(3))]
        first=project_pose(self.local,np.zeros(3),self.rig,np.zeros(3),contacts)
        second=project_pose(self.local,np.zeros(3),self.rig,np.zeros(3),contacts)
        np.testing.assert_array_equal(first[0],second[0])
        np.testing.assert_array_equal(first[1],second[1])

    def test_failure_selection_uses_measured_extremum_not_specific_asset_frames(self):
        count=31
        local=np.repeat(self.local[None],count,axis=0)
        root=np.zeros((count,3))
        reference=complete(local,root,self.rig)
        candidate_root=root.copy();candidate_root[7,1]=.4
        candidate=complete(local,candidate_root,self.rig)
        spec=dict(hands=('left',),contactSeconds=(.3,.6),groundedSeconds=(.7,.8))
        selected=select_failure_keys(candidate,reference,self.rig,spec,1.)
        self.assertEqual([item['frame'] for item in selected],[7])
        self.assertEqual(select_failure_keys(candidate,reference,self.rig,spec,1.,excluded={7}),[])

    def test_separate_pelvis_failure_intervals_each_produce_a_key(self):
        count=31;local=np.repeat(self.local[None],count,axis=0);root=np.zeros((count,3))
        reference=complete(local,root,self.rig)
        candidate_root=root.copy();candidate_root[7,1]=.4;candidate_root[20,1]=.3
        candidate=complete(local,candidate_root,self.rig)
        spec=dict(hands=('left',),contactSeconds=(.3,.6),groundedSeconds=(.7,.8))
        selected=select_failure_keys(candidate,reference,self.rig,spec,1.)
        self.assertEqual([item['frame'] for item in selected],[7,20])

    def test_unreachable_shoulder_is_detected_even_when_wrist_shift_fits_repair_budget(self):
        count=31;local=np.repeat(self.local[None],count,axis=0);root=np.zeros((count,3))
        reference=complete(local,root,self.rig)
        candidate_root=root.copy();candidate_root[12,0]=.02
        candidate=complete(local,candidate_root,self.rig)
        spec=dict(hands=('left',),contactSeconds=(.3,.6),groundedSeconds=(.7,.8))
        selected=select_failure_keys(candidate,reference,self.rig,spec,1.)
        self.assertEqual([item['frame'] for item in selected],[12])
        self.assertEqual(selected[0]['failures'][0]['metric'],'left-shoulder-reach')

    def test_landing_foot_orientation_can_fail_even_when_every_position_is_correct(self):
        count=31;local=np.repeat(self.local[None],count,axis=0);root=np.zeros((count,3))
        reference=complete(local,root,self.rig)
        foot=self.rig['jointNames'].index('LeftFoot')
        local=local.copy()
        local[26,foot]=Rotation.from_euler('y',45,degrees=True).as_matrix()
        candidate=complete(local,root,self.rig)
        spec=dict(hands=('left',),contactSeconds=(.3,.6),groundedSeconds=(.7,.8))
        selected=select_failure_keys(candidate,reference,self.rig,spec,1.)
        self.assertEqual([item['frame'] for item in selected],[26])
        self.assertEqual(selected[0]['failures'][0]['metric'],'left-landing-foot-orientation')

    def test_explicit_projection_tube_and_reach_reserve_are_constraints_not_new_acceptance(self):
        contacts=[dict(kind='hand',side='left',position=[-.3,.8,.6],rotation=np.eye(3),reachMarginMeters=.06)]
        result,root,receipt=project_pose(self.local,np.zeros(3),self.rig,np.zeros(3),contacts,projection_tube_meters=.11)
        self.assertLessEqual(np.linalg.norm(root),.11+1e-7)
        self.assertEqual(receipt['rootGateMeters'],.15)
        _,points=fk(result[None],root[None],self.rig)
        arm=self.rig['jointNames'].index('LeftArm')
        self.assertLessEqual(np.linalg.norm(points[0,arm]-contacts[0]['position']),.44+1e-7)
        self.assertEqual(receipt['contacts'][0]['projectedReachReserveCm'],6.)

    def test_invalid_tube_or_rotation_cannot_silently_pass(self):
        with self.assertRaises(ValueError):
            project_pose(self.local,np.zeros(3),self.rig,np.zeros(3),[],root_tube_meters=.2)
        with self.assertRaises(ValueError):
            project_pose(self.local,np.zeros(3),self.rig,np.zeros(3),[
                dict(kind='hand',side='left',position=[0,0,0],rotation=np.zeros((3,3)))])

    def test_feedback_updates_overlapping_soft_hip_key_and_leaves_original_inputs(self):
        count=31;local=np.repeat(self.local[None],count,axis=0);root=np.zeros((count,3))
        reference=complete(local,root,self.rig)
        candidate_root=root.copy();candidate_root[7,1]=.4
        candidate=complete(local,candidate_root,self.rig)
        rotations=Rotation.from_matrix(self.local).as_rotvec().tolist()
        conditions=[dict(type='fullbody',frame_indices=[0,30],local_joints_rot=[rotations,rotations],root_positions=[[0,0,0],[0,0,0]]),
                    dict(type='end-effector',joint_names=['Hips'],frame_indices=[7],local_joints_rot=[rotations],root_positions=[[0,0,0]])]
        before=copy.deepcopy(conditions)
        spec=dict(hands=('left',),contactSeconds=(.3,.6),groundedSeconds=(.7,.8))
        with patch('tools.kimodo_pipeline.mantle.profile',return_value=spec), patch(
            'tools.kimodo_pipeline.mantle.reference_input',return_value=(dict(playLength=1),{},{})):
            result,receipt=build_feedback_constraints(dict(semantic='Controlled',seconds=count/30,fps=30),
                candidate,reference,self.rig,conditions)
        self.assertEqual(conditions,before)
        self.assertEqual(result[0]['frame_indices'],[0,7,30])
        self.assertEqual(result[0]['root_positions'][1],result[1]['root_positions'][0])
        self.assertEqual(result[0]['local_joints_rot'][1],result[1]['local_joints_rot'][0])
        self.assertFalse(receipt['referenceFullBodyPoseCopied'])
        self.assertFalse(receipt['deliverable'])
        self.assertEqual(receipt['totalKeyUnion'],[0,7,30])

    def test_feedback_cannot_silently_exceed_sparse_union_budget(self):
        count=31;local=np.repeat(self.local[None],count,axis=0);root=np.zeros((count,3))
        reference=complete(local,root,self.rig)
        candidate_root=root.copy();candidate_root[7,1]=.4
        candidate=complete(local,candidate_root,self.rig)
        rotations=Rotation.from_matrix(self.local).as_rotvec().tolist()
        occupied=[frame for frame in range(20) if frame!=7]+[30]
        conditions=[dict(type='fullbody',frame_indices=[0,30],local_joints_rot=[rotations,rotations],root_positions=[[0,0,0],[0,0,0]]),
                    dict(type='end-effector',joint_names=['Hips'],frame_indices=occupied,
                         local_joints_rot=[rotations]*len(occupied),root_positions=[[0,0,0]]*len(occupied))]
        spec=dict(hands=('left',),contactSeconds=(.3,.6),groundedSeconds=(.7,.8))
        with patch('tools.kimodo_pipeline.mantle.profile',return_value=spec), patch(
            'tools.kimodo_pipeline.mantle.reference_input',return_value=(dict(playLength=1),{},{})), self.assertRaisesRegex(ProjectionError,'budget'):
            build_feedback_constraints(dict(semantic='Controlled',seconds=count/30,fps=30),candidate,reference,self.rig,conditions)

    def test_existing_soft_event_key_is_upgraded_without_growing_time_union(self):
        count=31;local=np.repeat(self.local[None],count,axis=0);root=np.zeros((count,3))
        reference=complete(local,root,self.rig)
        candidate_root=root.copy();candidate_root[7,1]=.16;candidate_root[6,1]=.155
        candidate=complete(local,candidate_root,self.rig)
        rotations=Rotation.from_matrix(self.local).as_rotvec().tolist()
        conditions=[dict(type='fullbody',frame_indices=[0,30],local_joints_rot=[rotations,rotations],root_positions=[[0,0,0],[0,0,0]]),
                    dict(type='end-effector',joint_names=['Hips'],frame_indices=[6],local_joints_rot=[rotations],root_positions=[[0,0,0]])]
        spec=dict(hands=('left',),contactSeconds=(.3,.6),groundedSeconds=(.7,.8))
        with patch('tools.kimodo_pipeline.mantle.profile',return_value=spec), patch(
            'tools.kimodo_pipeline.mantle.reference_input',return_value=(dict(playLength=1),{},{})):
            result,receipt=build_feedback_constraints(dict(semantic='Controlled',seconds=count/30,fps=30),candidate,reference,self.rig,
                conditions,reuse_existing_time_keys=True)
        self.assertEqual(result[0]['frame_indices'],[0,6,30])
        self.assertEqual(receipt['totalKeyUnion'],[0,6,30])
        self.assertEqual(receipt['selected'][0]['measuredFailureFrames'],[7])


if __name__=='__main__':
    unittest.main()

"""Check the actual FK targets emitted for Kimodo, not only JSON structure."""
from __future__ import annotations

import copy
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .gait_constraints import (GaitConstraintError, materialize_gait_constraints,
                               validate_pelvis_constraint_consistency)
from .motion import complete, fk
from .test_gait import walking_fixture


def parameters(speed=.6):
    return {'seconds': 5., 'fps': 60, 'targetPeriodSeconds': 1.,
            'rootPath': {'kind': 'linear', 'velocity': speed, 'direction': [0., 1.], 'headingDegrees': 0}}


def decode(constraint, rig):
    frames = constraint['frame_indices']
    axis = np.asarray(constraint['local_joints_rot'])
    local = Rotation.from_rotvec(axis.reshape(-1, 3)).as_matrix().reshape(len(frames), -1, 3, 3)
    root = np.asarray(constraint['root_positions'])
    global_rot, positions = fk(local, root, rig)
    return {frame: (local[index], root[index], global_rot[index], positions[index])
            for index, frame in enumerate(frames)}


class GaitConstraintTests(unittest.TestCase):
    def test_feet_only_conditions_preserve_measured_phase_and_support_without_body_pins(self):
        source, rig = walking_fixture(seconds=5.)
        params = parameters()
        params.update(seconds=3.02, maximumConstraintCycles=3, poseConstraintMode='feet_only',
                      fullBodyPhaseFractions=[.5], fullBodyPhaseCycles=3,
                      supportAnchorCount=2, preserveReferenceHeading=True)
        constraints, receipt = materialize_gait_constraints(params, source, rig)
        self.assertTrue(receipt['pass'])
        self.assertFalse(receipt['upperBodyConstrained'])
        self.assertEqual({'root2d', 'left-foot', 'right-foot'}, {c['type'] for c in constraints})
        for side in ('left', 'right'):
            constraint = next(c for c in constraints if c['type'] == side+'-foot')
            decoded = decode(constraint, rig)
            self.assertTrue(set(receipt['boundaryFrames']).issubset(decoded))
            self.assertLess(len(decoded), 20)
            foot = rig['jointNames'].index(side.title()+'Foot')
            for pair in receipt['supportPairs']:
                if pair['side'] != side:
                    continue
                for frame in pair['frames']:
                    np.testing.assert_allclose(decoded[frame][3][foot], pair['anchorMeters'], atol=1e-9)
                    np.testing.assert_allclose(decoded[frame][2][foot], pair['footGlobalRotation'], atol=1e-9)
        params['constrainHandOrientations'] = True
        with self.assertRaisesRegex(ValueError, 'cannot also pin the hands'):
            materialize_gait_constraints(params, source, rig)
        params['constrainHandOrientations'] = False
        params['poseConstraintMode'] = 'lower_body'
        lower_constraints, lower_receipt = materialize_gait_constraints(params, source, rig)
        self.assertFalse(lower_receipt['upperBodyConstrained'])
        self.assertEqual({'root2d', 'left-foot', 'right-foot', 'end-effector'},
                         {c['type'] for c in lower_constraints})
        hips = next(c for c in lower_constraints if c['type'] == 'end-effector')
        self.assertEqual(['Hips'], hips['joint_names'])
        self.assertEqual(receipt['boundaryFrames'], hips['frame_indices'])
        params['poseConstraintMode'] = 'ignore_contact'
        with self.assertRaisesRegex(ValueError, 'Unknown gait pose constraint mode'):
            materialize_gait_constraints(params, source, rig)

    def test_preserved_reference_heading_keeps_oblique_planted_leg_geometry(self):
        source, rig = walking_fixture(seconds=5.)
        rotation = Rotation.from_euler('y', 25, degrees=True).as_matrix()
        local = source['local_rot_mats'].copy()
        local[:, 0] = rotation @ local[:, 0]
        source = complete(local, source['root_positions'] @ rotation.T, rig, 60)
        params = parameters()
        params['rootPath']['direction'] = (rotation @ np.array([0., 0., 1.]))[[0, 2]].tolist()
        params.update(preserveReferenceHeading=True, matchStrideToPeriod=True,
                      fullBodyPhaseFractions=[.25, .5, .75])
        constraints, receipt = materialize_gait_constraints(params, source, rig)
        first = receipt['selectedReferenceCycle']['start']
        pinned_frames = {frame for pair in receipt['supportPairs'] for frame in pair['frames']}
        body = decode(next(c for c in constraints if c['type'] == 'fullbody'), rig)
        for frame, (_, root, _, joints) in body.items():
            if frame in pinned_frames:
                continue
            expected = source['posed_joints'][first + frame % 60]
            translation = root - source['root_positions'][first + frame % 60]
            np.testing.assert_allclose(joints, expected + translation, atol=1e-9)
        for pair in receipt['supportPairs']:
            self.assertLess(pair['maximumPositionErrorMeters'], 1e-9)
        self.assertLess(receipt['maximumHeadingErrorRadians'], 1e-10)
        self.assertLess(max(receipt['constraintCounts'].values()), 20)
        # The original constant-heading mode would rotate these legs away
        # from the direction of their planted world-space anchors.
        params['preserveReferenceHeading'] = False
        fixed, _ = materialize_gait_constraints(params, source, rig)
        fixed_body = decode(next(c for c in fixed if c['type'] == 'fullbody'), rig)
        self.assertGreater(np.linalg.norm(body[15][3][3] - fixed_body[15][3][3]), .04)

    def test_phase_varying_reference_heading_matches_every_sparse_constraint(self):
        source, rig = walking_fixture(seconds=5.)
        local = source['local_rot_mats'].copy()
        yaw = .02 * np.sin(np.arange(len(local)) * 2 * np.pi / 60)
        local[:, 0] = Rotation.from_euler('y', yaw[:, None]).as_matrix() @ local[:, 0]
        source = complete(local, source['root_positions'], rig, 60)
        params = parameters()
        params.update(preserveReferenceHeading=True, fullBodyPhaseFractions=[.25, .5, .75])
        constraints, receipt = materialize_gait_constraints(params, source, rig)
        headings = np.asarray(constraints[0]['global_root_heading'])
        self.assertGreater(np.ptp(np.arctan2(headings[:, 1], headings[:, 0])), .03)
        for constraint in constraints[1:]:
            for frame, (_, _, _, joints) in decode(constraint, rig).items():
                line = joints[5] - joints[1]
                observed = np.arctan2(line[2], -line[0])
                expected = np.arctan2(headings[frame, 1], headings[frame, 0])
                self.assertAlmostEqual(observed, expected, places=10)
        np.testing.assert_allclose(headings[:60], headings[60:120], atol=1e-12)
        params['preserveReferenceHeading'] = 'true'
        with self.assertRaisesRegex(ValueError, 'must be a boolean'):
            materialize_gait_constraints(params, source, rig)

    def test_swing_foot_seam_orientation_is_consistent_with_fullbody_and_sparse_budget(self):
        source,rig=walking_fixture(seconds=5.)
        params=parameters();params.update(seconds=3.02,footBoundaryContextFrames=1,
            boundaryContextFrames=1,constrainClipEndpoint=True,supportAnchorCount=3,
            fullBodyPhaseFractions=[1/60,59/60],fullBodyPhaseCycles=3)
        constraints,receipt=materialize_gait_constraints(params,source,rig)
        body=decode(next(c for c in constraints if c['type']=='fullbody'),rig)
        self.assertIn(59,receipt['footBoundaryFrames'])
        for side in ('left','right'):
            foot=rig['jointNames'].index(side.title()+'Foot')
            poses=decode(next(c for c in constraints if c['type']==side+'-foot'),rig)
            self.assertLess(len(poses),20)
            for frame in receipt['footBoundaryFrames']:
                np.testing.assert_allclose(poses[frame][2][foot],body[frame][2][foot],atol=1e-12)
                np.testing.assert_allclose(poses[frame][3][foot],body[frame][3][foot],atol=1e-12)
        params['footBoundaryContextFrames']=8
        with self.assertRaisesRegex(GaitConstraintError,'frame budget'):
            materialize_gait_constraints(params,source,rig)
        params['footBoundaryContextFrames']=True
        with self.assertRaisesRegex(ValueError,'Foot boundary context'):
            materialize_gait_constraints(params,source,rig)

    def test_hand_endpoint_orientation_targets_repeat_with_the_body_cycle(self):
        source,rig=walking_fixture(seconds=5)
        rig=copy.deepcopy(rig)
        for label,x in (('Left',.3),('Right',-.3)):
            rig['jointNames'].append(label+'Hand');rig['parents'].append(0);rig['offsets'].append([x,.4,.1])
        local=np.concatenate([source['local_rot_mats'],np.tile(np.eye(3),(len(source['root_positions']),2,1,1))],axis=1)
        local[:,-2]=Rotation.from_euler('z',(.2*np.sin(np.arange(len(local))/60*2*np.pi))[:,None]).as_matrix()
        source=complete(local,source['root_positions'],rig,60)
        params=parameters();params.update(seconds=3.02,fullBodyPhaseFractions=[.25,.5,.75],
            fullBodyPhaseCycles=3,constrainHandOrientations=True)
        constraints,receipt=materialize_gait_constraints(params,source,rig)
        for side in ('left','right'):
            poses=decode(next(c for c in constraints if c['type']==side+'-hand'),rig)
            hand=rig['jointNames'].index(side.title()+'Hand')
            for a,b in ((0,60),(15,75),(30,90)):
                np.testing.assert_allclose(poses[a][2][hand],poses[b][2][hand],atol=1e-10)
            self.assertLess(receipt['constraintCounts'][side+'-hand'],20)

    def test_repeated_phase_poses_and_support_interiors_are_measured(self):
        source,rig=walking_fixture(seconds=5)
        params=parameters();params.update(seconds=3.02,fullBodyPhaseFractions=[.25,.5,.75],
            fullBodyPhaseCycles=3,supportAnchorCount=4,maximumAnchorCorrectionMeters=.15)
        constraints,receipt=materialize_gait_constraints(params,source,rig)
        self.assertTrue(receipt['pass'])
        poses=decode(next(c for c in constraints if c['type']=='fullbody'),rig)
        for phase in (15,30,45):
            np.testing.assert_allclose(poses[phase][0],poses[phase+60][0],atol=1e-12)
        for pair in receipt['supportPairs']:
            self.assertEqual(4,len(pair['frames']))
            decoded=decode(next(c for c in constraints if c['type']==pair['side']+'-foot'),rig)
            foot=rig['jointNames'].index(pair['side'].title()+'Foot')
            for frame in pair['frames']:np.testing.assert_allclose(decoded[frame][3][foot],pair['anchorMeters'],atol=1e-10)
        self.assertLess(max(receipt['constraintCounts'].values()),20)

    def test_boundary_neighborhoods_repeat_position_and_motion_context(self):
        motion, rig = walking_fixture(seconds=5.)
        params = parameters(); params['boundaryContextFrames'] = 1
        constraints, receipt = materialize_gait_constraints(params, motion, rig)
        fullbody = next(item for item in constraints if item['type'] == 'fullbody')
        poses = decode(fullbody, rig)
        self.assertLess(len(poses), 20)
        for offset in (-1, 0, 1):
            a, b = poses[60+offset], poses[120+offset]
            np.testing.assert_allclose(a[0], b[0], atol=1e-12)
            np.testing.assert_allclose(b[3]-a[3], np.tile([0., 0., .6], (len(rig['jointNames']), 1)), atol=1e-12)
        before = poses[60][3]-poses[59][3]
        after = poses[120][3]-poses[119][3]
        np.testing.assert_allclose(before, after, atol=1e-12)
        self.assertTrue(receipt['pass'])

    def test_support_anchors_are_world_fixed_while_hips_advance(self):
        motion, rig = walking_fixture(seconds=5.)
        constraints, receipt = materialize_gait_constraints(parameters(), motion, rig)
        self.assertTrue(receipt['pass'])
        self.assertFalse(receipt['referenceAcceptedAsMaster'])
        by_type = {item['type']: item for item in constraints}
        self.assertEqual(set(by_type), {'root2d', 'fullbody', 'left-foot', 'right-foot'})
        for kind in ('fullbody', 'left-foot', 'right-foot'):
            self.assertLess(len(by_type[kind]['frame_indices']), 20)
        for pair in receipt['supportPairs']:
            decoded = decode(by_type[pair['side'] + '-foot'], rig)
            foot = rig['jointNames'].index(pair['side'].title() + 'Foot')
            toe = rig['jointNames'].index(pair['side'].title() + 'ToeEnd')
            first, last = [decoded[frame] for frame in pair['frames']]
            self.assertGreater(last[1][2] - first[1][2], .01)
            np.testing.assert_allclose(first[3][foot], last[3][foot], atol=1e-10)
            np.testing.assert_allclose(first[3][toe], last[3][toe], atol=1e-10)
            np.testing.assert_allclose(first[2][foot], last[2][foot], atol=1e-10)
            np.testing.assert_allclose(first[3][foot], pair['anchorMeters'], atol=1e-10)

    def test_fullbody_cycle_boundaries_repeat_pose_plus_actual_travel(self):
        motion, rig = walking_fixture(seconds=5.)
        constraints, _ = materialize_gait_constraints(parameters(), motion, rig)
        fullbody = next(item for item in constraints if item['type'] == 'fullbody')
        poses = decode(fullbody, rig)
        frames = fullbody['frame_indices']
        for start, end in zip(frames, frames[1:]):
            np.testing.assert_allclose(poses[start][0], poses[end][0], atol=1e-12)
            np.testing.assert_allclose(poses[end][3] - poses[start][3],
                                       np.tile([0., 0., .6], (len(rig['jointNames']), 1)), atol=1e-12)

    def test_all_constraint_roots_and_headings_match_dense_root2d(self):
        motion, rig = walking_fixture(seconds=5.)
        params = parameters(); params['rootPath']['headingDegrees'] = 37
        params['rootPath']['direction'] = [.6, .8]
        constraints, receipt = materialize_gait_constraints(params, motion, rig)
        path = np.asarray(constraints[0]['smooth_root_2d'])
        for constraint in constraints[1:]:
            for frame, (_, root, _, joints) in decode(constraint, rig).items():
                np.testing.assert_allclose(root[[0, 2]], path[frame], atol=1e-12)
                hip_line = joints[5] - joints[1]
                self.assertAlmostEqual(np.arctan2(hip_line[2], -hip_line[0]), np.deg2rad(37), places=10)
        self.assertLess(receipt['maximumHeadingErrorRadians'], 1e-10)

    def test_reference_rejection_is_not_relabelled_master_acceptance(self):
        motion, rig = walking_fixture(seconds=1.5)
        original = copy.deepcopy(motion)
        constraints, receipt = materialize_gait_constraints(parameters(), motion, rig)
        self.assertTrue(constraints)
        self.assertFalse(receipt['referenceAuditPass'])
        self.assertFalse(receipt['referenceAcceptedAsMaster'])
        np.testing.assert_array_equal(motion['root_positions'], original['root_positions'])
        np.testing.assert_array_equal(motion['local_rot_mats'], original['local_rot_mats'])

    def test_unreachable_fixed_foot_anchor_is_a_failure(self):
        motion, rig = walking_fixture(seconds=5.)
        with self.assertRaises(GaitConstraintError) as failure:
            materialize_gait_constraints(parameters(speed=30.), motion, rig)
        self.assertIn('No reachable support pair', str(failure.exception))
        self.assertFalse(failure.exception.receipt['pass'])
        self.assertFalse(failure.exception.receipt['referenceAcceptedAsMaster'])

    def test_missing_observed_cycle_cannot_be_replaced_with_percent_crop(self):
        motion, rig = walking_fixture(seconds=5.)
        motion['root_positions'][:, 1] += 2
        with self.assertRaisesRegex(GaitConstraintError, 'no complete alternating'):
            materialize_gait_constraints(parameters(), motion, rig)

    def test_shared_hip_lowering_is_explicit_and_never_exceeds_configured_limit(self):
        motion, rig = walking_fixture(seconds=5.)
        params = parameters(speed=10.)
        constraints, receipt = materialize_gait_constraints(params, motion, rig)
        maximum_drop = max(item['dropMeters'] for item in receipt['hipHeightAdjustments'])
        self.assertGreater(maximum_drop, .005)
        self.assertLessEqual(maximum_drop, .03)
        by_frame = {}
        for constraint in constraints[1:]:
            for frame, (_, root, _, _) in decode(constraint, rig).items():
                if frame in by_frame:
                    np.testing.assert_allclose(root, by_frame[frame], atol=1e-12)
                by_frame[frame] = root
        params['maximumHipLoweringMeters'] = .005
        with self.assertRaisesRegex(GaitConstraintError, 'hip-lowering limit'):
            materialize_gait_constraints(params, motion, rig)

    def test_sparse_frame_limit_is_enforced_before_generation(self):
        motion, rig = walking_fixture(seconds=5.)
        params = parameters(); params['seconds'] = 15
        with self.assertRaisesRegex(GaitConstraintError, 'frame budget'):
            materialize_gait_constraints(params, motion, rig)

    def test_nonunit_path_and_subframe_cycle_are_rejected(self):
        motion, rig = walking_fixture()
        params = parameters(); params['rootPath']['direction'] = [0., 2.]
        with self.assertRaisesRegex(ValueError, 'unit XZ'):
            materialize_gait_constraints(params, motion, rig)
        params = parameters(); params['targetPeriodSeconds'] = 1.001
        with self.assertRaisesRegex(ValueError, 'integral'):
            materialize_gait_constraints(params, motion, rig)


class NativePelvisConstraintTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from .als_reference import native_gait_reference
        from .constraint_phase import build_native_phase_evidence
        from .graph import read_json
        from .paths import project_root, reference_root
        receipts=read_json(project_root()/'artifacts/kimodo_als/nonprop-regeneration-20261009-v1/combined-receipts.json')
        cls.rig=read_json(receipts['rig:soma']['outputs']['rig.json']['path'])
        index_path=reference_root()/'assets/config/refactored_animation_sources.json'
        source=next(row['source'] for row in read_json(index_path)['assets'] if row['source'].endswith('.A_Als_Sprint_Acceleration'))
        cls.motion,_=native_gait_reference(index_path,source,cls.rig,5.,[0.,1.])
        cls.evidence=build_native_phase_evidence(index_path,source,cls.motion,cls.rig)
        cls.params=dict(seconds=2.,fps=30,targetPeriodSeconds=.6,rootPath=dict(kind='linear',velocity=5.,direction=[0.,1.]),
            preserveReferencePelvisPath=True,preserveReferenceHeading=True,maximumConstraintCycles=3,
            supportAnchorCount=2,fullBodyPhaseFractions=[],boundaryContextFrames=0,constrainClipEndpoint=True)

    def make(self, **updates):
        return materialize_gait_constraints(dict(self.params,**updates),self.motion,self.rig,phase_evidence=self.evidence)

    def check_layout(self,constraints):
        original,_=self.make()
        root=original[0];count=len(root['frame_indices'])
        hips=next(item for item in original if item['type']=='end-effector')
        # Independently derive expected residual from bound source, preserving
        # the selected native phase. Do not take the candidate's own claims.
        _,receipt=self.make();start=receipt['selectedReferenceCycle']['start']
        ids=start+np.arange(count)%18
        smooth=np.asarray(root['smooth_root_2d'])
        residual=self.motion['root_positions'][ids][:,[0,2]]-ids[:,None]/30*np.array([0.,5.])
        directions=np.asarray(root['global_root_heading']);headings=np.arctan2(directions[:,1],directions[:,0])
        return validate_pelvis_constraint_consistency(constraints,self.rig,smooth,residual,headings)

    def test_preserves_actual_pelvis_residual_and_identical_fullbody_foot_payloads(self):
        constraints,receipt=self.make()
        self.assertEqual(receipt['maximumPathErrorMeters'],0.)
        self.assertLess(receipt['maximumPelvisResidualErrorMeters'],1e-12)
        self.assertGreater(receipt['nativePelvisPath']['maximumResidualMeters'],.09)
        self.assertFalse(receipt['referenceAcceptedAsMaster'])
        self.assertFalse(receipt['referenceAuditPass'])
        self.assertLess(max(receipt['constraintCounts'].values()),20)
        self.assertEqual(constraints[-1]['type'],'end-effector')
        self.assertEqual(constraints[-1]['joint_names'],['Hips'])
        self.assertLess(self.check_layout(constraints)['maximumSharedFramePositionErrorMeters'],1e-12)

    def test_requested_cycles_exclude_the_closing_boundary_as_another_start(self):
        for count in (2,3):
            constraints,receipt=self.make(maximumConstraintCycles=count)
            self.assertEqual(receipt['completeConstrainedCycles'],count)
            self.assertEqual(receipt['cycleStartFrames'],[18*i for i in range(count)])
            self.assertEqual(len(receipt['supportPairs']),2*count)
            self.assertTrue(all(pair['cycleBoundaryFrame']<18*count for pair in receipt['supportPairs']))
        with self.assertRaisesRegex(GaitConstraintError,'do not fit'):
            self.make(maximumConstraintCycles=4)

    def test_union_budget_is_enforced_even_when_individual_feet_fit(self):
        with self.assertRaisesRegex(GaitConstraintError,'frame budget') as failed:
            self.make(maxConstraintFramesPerType=12)
        counts=failed.exception.receipt['constraintCounts']
        self.assertLessEqual(counts['left-foot'],12)
        self.assertLessEqual(counts['right-foot'],12)
        self.assertGreater(counts['end-effector'],12)

    def test_new_mode_rejects_unverified_or_incompatible_controls(self):
        with self.assertRaisesRegex(ValueError,'verified native phase'):
            materialize_gait_constraints(self.params,self.motion,self.rig)
        for updates in (dict(preserveReferencePelvisPath=1),dict(preserveReferenceHeading=False),
                        dict(constrainHandOrientations=True),dict(pelvisConstraintBackend='another-backend')):
            with self.subTest(updates=updates):
                with self.assertRaises(ValueError):self.make(**updates)

    def test_same_frame_conflict_cannot_be_hidden_by_backend_overwrite_order(self):
        constraints,_=self.make()
        bad=copy.deepcopy(constraints)
        # Changing only a local joint preserves the declared Hips residual but
        # contradicts the other payloads at the same sparse frame.
        hips=bad[-1]
        joint=self.rig['jointNames'].index('LeftForeArm')
        hips['local_joints_rot'][0][joint][0]+=.1
        with self.assertRaisesRegex(ValueError,'Conflicting sparse constraint poses'):
            self.check_layout(bad)
        bad=copy.deepcopy(constraints);bad[-1]['root_positions'][0][0]+=.001
        with self.assertRaisesRegex(ValueError,'Hips residual'):
            self.check_layout(bad)
        bad=copy.deepcopy(constraints);bad[0]['smooth_root_2d'][0][0]+=.001
        with self.assertRaisesRegex(ValueError,'requested trajectory'):
            self.check_layout(bad)
        bad=copy.deepcopy(constraints);bad.insert(1,bad.pop())
        with self.assertRaisesRegex(ValueError,'named feet followed'):
            self.check_layout(bad)

    def test_real_kimodo_loader_and_postprocess_extraction_preserve_hips_and_named_feet(self):
        from .paths import configured_path, PIPELINE_ROOT
        constraints,_=self.make(poseConstraintMode='feet_only')
        # Execute the actual installed backend in its configured Python rather
        # than emulating its dtype, load, hierarchy or postprocess behavior.
        script='''
import copy,json,sys,torch
sys.path.insert(0,sys.argv[2])
from kimodo.constraints import load_constraints_lst,EndEffectorConstraintSet
from kimodo.geometry import quaternion_to_matrix
from kimodo.postprocess import extract_input_motion_from_constraints
from kimodo.skeleton import SOMASkeleton77,SOMASkeleton30
torch.set_num_threads(2)
payload=json.load(open(sys.argv[1],encoding='utf-8'))
count=len(payload[0]['frame_indices'])
for Skeleton in (SOMASkeleton77,SOMASkeleton30):
    skeleton=Skeleton().to(device='cpu')
    loaded=load_constraints_lst(payload,skeleton)
    expected={id(item):item.global_joints_positions.clone() for item in loaded if isinstance(item,EndEffectorConstraintSet)}
    smooth=loaded[0].smooth_root_2d.clone()
    hips,quaternions=extract_input_motion_from_constraints(loaded,skeleton,count,skeleton.nbjoints)
    _,actual,_=skeleton.fk(quaternion_to_matrix(quaternions),hips)
    assert torch.equal(loaded[0].smooth_root_2d,smooth)
    for item in loaded:
        if isinstance(item,EndEffectorConstraintSet):
            if item.joint_names==['Hips']:
                torch.testing.assert_close(hips[item.frame_indices],expected[id(item)][:,skeleton.root_idx],rtol=0,atol=0)
            else:
                torch.testing.assert_close(actual[item.frame_indices][:,item.pos_indices],expected[id(item)][:,item.pos_indices])
    bad=copy.deepcopy(payload);bad.insert(1,bad.pop())
    reordered=load_constraints_lst(bad,skeleton)
    desired=reordered[1].global_joints_positions[:,skeleton.root_idx].clone()
    wrong,_=extract_input_motion_from_constraints(reordered,skeleton,count,skeleton.nbjoints)
    assert torch.linalg.vector_norm(wrong[reordered[1].frame_indices]-desired,dim=-1).max()>.01
print('KIMODO_PELVIS_AND_FOOT_EXTRACTION_VERIFIED')
'''
        with tempfile.TemporaryDirectory() as directory:
            payload=Path(directory)/'constraints.json'
            payload.write_text(json.dumps(constraints),encoding='utf-8')
            result=subprocess.run([str(configured_path('python')),'-c',script,str(payload),str(configured_path('kimodo_repo'))],
                cwd=PIPELINE_ROOT,capture_output=True,text=True,timeout=120)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        self.assertIn('KIMODO_PELVIS_AND_FOOT_EXTRACTION_VERIFIED',result.stdout)


if __name__ == '__main__':
    unittest.main()

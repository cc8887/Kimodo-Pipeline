"""Check extraction, travel, interpolation boundaries and refusal of bad loops."""
import unittest
import tempfile
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

from . import motion,gait
from .loop_segment import crop_loop,tile_loop


class LoopSegmentTests(unittest.TestCase):
    def fixture(self):
        from .test_gait import walking_fixture
        _,rig=walking_fixture()
        t=np.arange(65)/30
        local=np.tile(np.eye(3),(65,len(rig['jointNames']),1,1))
        local[:,0]=Rotation.from_euler('z',(2*np.sin(2*np.pi*t))[:,None],degrees=True).as_matrix()
        root=np.column_stack([.01*np.sin(2*np.pi*t),1+.005*np.cos(2*np.pi*t),.6*t])
        return motion.complete(local,root,rig,30),rig

    def test_single_inclusive_loop_and_exact_travel_reconstruction(self):
        source,rig=self.fixture()
        loop,report=crop_loop(source,rig,15,45,close=False)
        self.assertEqual(31,len(loop['root_positions']))
        self.assertEqual(30,report['uniqueKeyCount'])
        self.assertEqual(1.,report['durationSeconds'])
        recovered=loop['root_positions']+report['removedPlanarTrajectory']['positions']
        np.testing.assert_allclose(recovered,source['root_positions'][15:46],atol=1e-14)
        np.testing.assert_array_equal(loop['local_rot_mats'],source['local_rot_mats'][15:46])
        self.assertFalse(report['productionReady'])
        self.assertFalse(report['stableGeneratedCyclesProven'])

    def test_geodesic_closure_keeps_interior_and_matches_rotation_velocity(self):
        source,rig=self.fixture()
        loop,report=crop_loop(source,rig,15,44,seam_frames=3)
        np.testing.assert_allclose(loop['local_rot_mats'][0],loop['local_rot_mats'][-1],atol=1e-13)
        np.testing.assert_array_equal(loop['local_rot_mats'][4:-4],source['local_rot_mats'][19:41])
        self.assertLess(report['after']['angularVelocityRps'],1e-12)
        self.assertEqual(0.,report['after']['bodyMeters'])
        self.assertFalse(report['productionReady'])

    def test_tile_restores_displacement_without_extra_duplicate_key(self):
        source,rig=self.fixture()
        loop,report=crop_loop(source,rig,15,45)
        displacement=np.asarray(report['removedPlanarTrajectory']['displacement'])
        tiled=tile_loop(loop,rig,cycles=4,displacement=displacement)
        self.assertEqual(121,len(tiled['root_positions']))
        for cycle in range(4):
            np.testing.assert_allclose(tiled['root_positions'][cycle*30],loop['root_positions'][0]+cycle*displacement)
        np.testing.assert_allclose(tiled['root_positions'][-1],loop['root_positions'][-1]+4*displacement)
        np.testing.assert_array_equal(source['root_positions'],self.fixture()[0]['root_positions'])

    def test_open_endpoint_cannot_be_hidden_by_tiling(self):
        source,rig=self.fixture()
        loop,_=crop_loop(source,rig,15,44,close=False)
        with self.assertRaisesRegex(ValueError,'open loop'):
            tile_loop(loop,rig,displacement=[0,0,.6])

    def test_original_correction_bounds_remain_enforced(self):
        source,rig=self.fixture()
        source['local_rot_mats'][45,0]=Rotation.from_euler('y',90,degrees=True).as_matrix()
        with self.assertRaisesRegex(ValueError,'correction budgets'):
            crop_loop(source,rig,15,45)

    def test_invalid_or_overlapping_crop_windows_are_rejected(self):
        source,rig=self.fixture()
        for start,end,window in [(-1,30,3),(0,65,3),(0,6,3),(0,30,1),(True,30,3),(0,30.,3)]:
            with self.subTest(start=start,end=end,window=window),self.assertRaises(ValueError):
                crop_loop(source,rig,start,end,seam_frames=window)

    def test_closed_pose_is_not_a_contact_certificate(self):
        from .test_gait import walking_fixture
        source,rig=walking_fixture(seconds=5.)
        # Preserve a real grounded sliding error; closure must not approve it.
        source['root_positions'][:,0]+=np.arange(len(source['root_positions']))/60*.5
        loop,report=crop_loop(source,rig,60,120)
        tiled=tile_loop(loop,rig,displacement=report['removedPlanarTrajectory']['displacement'])
        audit=gait.audit_candidate(tiled,rig)
        self.assertEqual(0.,report['after']['bodyMeters'])
        self.assertFalse(audit['pass'])

    def test_positive_grounded_cycle_passes_unchanged_playback_gate(self):
        from .test_gait import walking_fixture
        source,rig=walking_fixture(seconds=5.)
        loop,report=crop_loop(source,rig,60,120)
        tiled=tile_loop(loop,rig,displacement=report['removedPlanarTrajectory']['displacement'])
        self.assertTrue(gait.audit_candidate(tiled,rig,.6)['pass'])
        self.assertFalse(report['stableGeneratedCyclesProven'])

    def test_optional_paths_keep_frame_and_bone_identity(self):
        source,rig=self.fixture()
        loop,report=crop_loop(source,rig,15,44,include_correction_paths=True)
        paths=report['correctionPaths']
        self.assertEqual((30,len(rig['jointNames'])),np.asarray(paths['rotationDegrees']).shape)
        self.assertEqual((30,2),np.asarray(paths['ankleMeters']).shape)
        self.assertEqual((30,),np.asarray(paths['rootMeters']).shape)
        self.assertEqual(report['correctionBudget']['maximumRotationDegrees'],np.asarray(paths['rotationDegrees']).max())

    def test_failed_crop_preserves_loop_and_blocks_dependent_master(self):
        from .runner import Runner,write_json
        from .batch import run_batch
        source,rig=self.fixture()
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            write_json(root/'rig.json',rig);motion.save_motion(root/'source.npz',source,rig)
            nodes={'rig:soma':dict(id='rig:soma',op='rig',deps=[],params={}),
                   'source':dict(id='source',op='external',deps=['rig:soma'],params={'reason':'fixture'}),
                   'crop':dict(id='crop',op='loop_crop',deps=['rig:soma','source'],
                       params=dict(source='source',startFrame=15,endFrameInclusive=45,expectedSpeedMps=.6)),
                   'master':dict(id='master',op='gait_cycle',deps=['rig:soma','crop'],
                       params=dict(source='crop',expectedSpeedMps=.6,targetCount=31,policy={}))}
            runner=Runner(dict(nodes=nodes,rules={'fps':30},rulesDigest='test'),root/'run',bindings={
                'rig:soma':{'path':str(root/'rig.json')},'source':{'path':str(root/'source.npz')}})
            result=run_batch(runner,['master'])
            self.assertEqual('failed',result['nodes']['crop']['status'])
            self.assertEqual('blocked',result['nodes']['master']['status'])
            self.assertEqual(0,result['completedTargets'])
            self.assertNotIn('crop',runner.results)
            self.assertEqual(1,len(list((root/'run').glob('.building-*/single-loop.npz'))))


if __name__=='__main__':unittest.main()

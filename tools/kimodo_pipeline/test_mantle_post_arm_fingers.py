"""Whole-clip real skin regression for arm-then-finger refinement with original budgets."""
import copy
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from .graph import read_json
from .paths import PIPELINE_ROOT
from .runner import Runner
from .mantle import reference_input,target,_hand_frames,profile
from .mantle_hand_repair import immutable_target_skin,repair,validate_report
from .offline_retarget import convert
from .publish import world_pose
from .root_ownership import semantic_hash


class MantlePostArmFingerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        native,cls.skeleton,_=reference_input('Mantle_High')
        cls.skin=immutable_target_skin(cls.skeleton);cls.samples=[]
        for style in ('default-sparse','cautious-sparse'):
            folder=PIPELINE_ROOT/'workspace/runs/mantle-generalization-20261013-v1/target-feedback-v2'/style/'round-6'
            graph=read_json(folder/'graph.json');runner=Runner(graph,folder/'authoring',None)
            runner.results=read_json(folder/'authoring/receipts.json')
            timing={key:native[key] for key in ('sampledKeyCount','playLength','frameRateNumerator','frameRateDenominator')}
            raw,_=convert(runner.motion('motion:Mantle_High'),runner.rig(),cls.skeleton,native['source'],timing)
            raw,_,_,gate=target(raw,cls.skeleton,'Mantle_High')
            if not gate['passed']:raise AssertionError('Fixture must retain the original 10cm target gate')
            before_hash=semantic_hash(raw)
            result,report=repair(raw,cls.skeleton,'Mantle_High',cls.skin,reconstruct_twist=True)
            cls.samples.append((style,raw,result,report,before_hash))

    def test_both_complete_clips_pass_without_resetting_original_budgets(self):
        for style,raw,result,report,before_hash in self.samples:
            with self.subTest(style=style):
                self.assertEqual(semantic_hash(raw),before_hash)
                self.assertTrue(report['passed'],[row for row in report['records'] if row['afterDepthCm']>2])
                validate_report(report)
                self.assertTrue(report['temporalSkin']['passed'])
                self.assertEqual(report['temporalSkin']['originalReferenceSha256'],before_hash)
                self.assertEqual(report['temporalSkin']['sampleCount'],3201)
                self.assertLessEqual(max(r['maximumDepthCm'] for r in report['temporalSkin']['hands']),2.)
                refinements=[item for row in report['records'] for item in row['postArmFingerRefinement']]
                self.assertTrue(refinements)
                self.assertTrue(any(item['achievedClearanceCm']>=.19 for item in refinements))
                self.assertTrue(all(item['wholeDepthAfterCm']<=item['wholeDepthBeforeCm']+1e-7 for item in refinements))
                by_name={track['bone']:track for track in result['tracks']}
                for old in raw['tracks']:
                    new=by_name[old['bone']]
                    self.assertEqual(old['positions'],new['positions'])
                    self.assertEqual(old['scales'],new['scales'])
                    if old['bone'].startswith(('thumb_','index_','middle_','ring_','pinky_')):
                        before=Rotation.from_quat(old['rotations']);after=Rotation.from_quat(new['rotations'])
                        self.assertLessEqual(np.max((before.inv()*after).magnitude())*180/np.pi,45.001)
                    elif not old['bone'].startswith(('upperarm','lowerarm','hand','thigh_twist','calf_twist')):
                        self.assertEqual(old,new)

    def test_world_wrist_orientation_support_and_total_translation_are_preserved(self):
        for style,raw,result,report,_ in self.samples:
            with self.subTest(style=style):
                before=world_pose(raw,self.skeleton);after=world_pose(result,self.skeleton)
                for side in ('left','right'):
                    name='hand_'+side[0]
                    displacement=np.linalg.norm(after[name][1]-before[name][1],axis=1)
                    self.assertLessEqual(float(displacement.max()),10.001)
                    frames=_hand_frames(profile('Mantle_High'),side,raw['sampledKeyCount'],raw['playLength'])
                    self.assertLess(float(displacement[frames].max()),1e-5)
                    angles=Rotation.from_matrix(before[name][0].swapaxes(-1,-2)@after[name][0]).magnitude()*180/np.pi
                    self.assertLess(float(angles.max()),1e-5)

    def test_new_receipt_rejects_missing_subframes_or_reset_cumulative_budget(self):
        report=copy.deepcopy(self.samples[0][3])
        del report['temporalSkin']
        with self.assertRaisesRegex(ValueError,'interpolation skin evidence'):validate_report(report)
        report=copy.deepcopy(self.samples[0][3])
        report['temporalSkin']['hands'][0]['maximumCumulativeWristCorrectionCm']=10.1
        with self.assertRaisesRegex(ValueError,'cumulative budget'):validate_report(report)


if __name__=='__main__':unittest.main()

"""Real accepted source motions must compile target-aware retry conditions."""
import copy
from pathlib import Path
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from .graph import digest,read_json
from .mantle_target_feedback import build_target_feedback_constraints
from .mantle_target_projection import TargetPoseObserver
from .mantle import reference_input
from .motion import fk
from .paths import PIPELINE_ROOT
from .runner import Runner


class MantleTargetFeedbackIntegrationTests(unittest.TestCase):
    def test_both_real_styles_build_without_altering_source_or_time_budget(self):
        root=PIPELINE_ROOT/'workspace/runs/mantle-generalization-20261013-v1/feedback-existing-key-v1'
        for style in ('default-sparse','cautious-sparse'):
            with self.subTest(style=style):
                folder=root/style/'round-5'
                graph=read_json(folder/'graph.json')
                runner=Runner(graph,folder/'authoring',None)
                runner.results=read_json(folder/'authoring/receipts.json')
                motion=runner.motion('motion:Mantle_High');reference=runner.motion('reference:Mantle_High:native-mantle')
                conditions=read_json(runner.output_of('candidate:Mantle_High')['constraints.json']['path'])
                original=copy.deepcopy(conditions);local=motion['local_rot_mats'].copy();root_positions=motion['root_positions'].copy()
                changed,report=build_target_feedback_constraints(graph['nodes']['candidate:Mantle_High']['params'],motion,reference,runner.rig(),conditions)
                self.assertTrue(report['passed']);self.assertFalse(report['deliverable'])
                self.assertEqual(report['constraintsSha256'],digest(changed))
                self.assertEqual(report['totalKeyUnion'],sorted({i for c in conditions for i in c['frame_indices']}))
                self.assertLess(len(report['totalKeyUnion']),20)
                self.assertEqual(conditions,original)
                np.testing.assert_array_equal(motion['local_rot_mats'],local)
                np.testing.assert_array_equal(motion['root_positions'],root_positions)
                self.assertTrue(all(item['productionRetargetParity']['passed'] for item in report['selected']))
                for item in report['selected']:
                    self.assertTrue(item['projection']['passed'])
                    self.assertTrue(all(row['projectedTargetCorrectionCm']<=10-item['projection']['conditioningMarginCm']+1e-5
                                        for row in item['projection']['contacts']))


if __name__=='__main__':unittest.main()

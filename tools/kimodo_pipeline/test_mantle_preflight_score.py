"""Failed subframe correction budgets must remain visible to bounded retries."""
import copy
import unittest
from .mantle_target_preflight import failure_score
from .graph import read_json
from .paths import PIPELINE_ROOT


class MantlePreflightScoreTests(unittest.TestCase):
    def fixture(self):
        return dict(checks=[dict(assessment=dict(checks=[dict(check='boundedTargetHandSkinRepair',passed=False)],
            handSkinRepair=dict(maximumDepthCm=2.,maximumArmCorrectionCm=10.,maximumFingerCorrectionDegrees=45.,records=[],
                temporalSkin=dict(hands=[dict(maximumDepthCm=1.,maximumCumulativeWristCorrectionCm=9.,
                    maximumCumulativeFingerCorrectionDegrees=40.,maximumFullSupportReachResidualCm=0.)]))))])

    def test_subframe_budgets_are_not_hidden_by_legal_skin_depth(self):
        for field,value,expected in [('maximumCumulativeWristCorrectionCm',12.,1.2),
                ('maximumCumulativeFingerCorrectionDegrees',54.,1.2),('maximumFullSupportReachResidualCm',.003,3.)]:
            with self.subTest(field=field):
                fixture=self.fixture();fixture['checks'][0]['assessment']['handSkinRepair']['temporalSkin']['hands'][0][field]=value
                self.assertAlmostEqual(failure_score(fixture),expected)

    def test_authored_skin_failure_still_competes_with_subframe_metrics(self):
        fixture=self.fixture();fixture['checks'][0]['assessment']['handSkinRepair']['records']=[dict(afterDepthCm=3.,transitionAnchorDepthCm=0.)]
        self.assertAlmostEqual(failure_score(fixture),1.5)

    def test_passed_preflight_has_zero_failure_score(self):
        fixture=self.fixture();fixture['checks'][0]['assessment']['checks'][0]['passed']=True
        self.assertEqual(failure_score(fixture),0.)

    def test_actual_delivered_report_shape_tracks_new_subframe_failure(self):
        base=PIPELINE_ROOT/'workspace/runs/mantle-generalization-20261013-v1/target-feedback-v2'
        for style in ('default-sparse','cautious-sparse'):
            with self.subTest(style=style):
                original=read_json(base/style/'round-6/continuations/run-002/completion-target-preflight.json')[0]
                self.assertTrue(original['passed'])
                self.assertEqual(failure_score(original),0.)
                failed=copy.deepcopy(original);assessment=failed['checks'][0]['assessment']
                skin=assessment['handSkinRepair'];skin['passed']=False;skin['temporalSkin']['passed']=False
                skin['temporalSkin']['hands'][0]['maximumCumulativeWristCorrectionCm']=12.
                next(c for c in assessment['checks'] if c['check']=='boundedTargetHandSkinRepair')['passed']=False
                self.assertAlmostEqual(failure_score(failed),1.2)


if __name__=='__main__':unittest.main()

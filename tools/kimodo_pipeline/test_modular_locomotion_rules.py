"""CLI planning tests; require the configured ALS contract snapshot, no model/GPU."""
import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from . import paths
from .graph import load_rules, read_json

ROOT = paths.PIPELINE_ROOT
RULES = ROOT / 'tools/kimodo_pipeline/modular_locomotion_rules.json'
MOTIONS = ('Sprint', 'Sprint_Acceleration', 'Walk_Backward')


class ModularLocomotionPlanTests(unittest.TestCase):
    @unittest.skipUnless(paths.contract_path().is_file(), 'Requires configured ALS contract snapshot')
    def test_preset_scope_is_bounded_and_cannot_select_an_unconfigured_motion(self):
        cli=[sys.executable,str(ROOT/'tools/produce_als_resources.py'),'inspect',
             '--scope','locomotion','--rules',str(RULES)]
        data=json.loads(subprocess.run(cli,cwd=ROOT,check=True,capture_output=True,text=True).stdout)
        self.assertEqual(sum(key.startswith('asset:') for key in data['selectedOrder']),6)
        bad=subprocess.run(cli+['--motion','Run_Forward'],cwd=ROOT,capture_output=True,text=True)
        self.assertNotEqual(bad.returncode,0)
        self.assertIn('outside the selected scope',bad.stderr)

    def test_portable_configuration_keeps_original_budgets(self):
        raw = read_json(RULES)
        self.assertEqual(raw['extends'], 'production_rules.json')
        self.assertEqual(raw['promptModules']['style'], 'feminine')
        self.assertNotIn('locomotionPosture', raw)
        for spec in raw['gaitClosure'].values():
            self.assertNotIn('candidateCycles', spec)  # Must not override actual 3 seconds.
            self.assertNotIn('upperBodyIdleReference', spec)
            self.assertFalse(any('budget' in k.lower() or 'maximumHip' in k or 'maxPose' in k for k in spec))
        text = RULES.read_text(encoding='utf-8')
        self.assertNotIn(':/', text)
        self.assertNotIn('textCache', text)

    @unittest.skipUnless(paths.contract_path().is_file(), 'Requires configured ALS contract snapshot')
    def test_cli_plan_and_inspect_all_styles_keep_six_deliveries_and_no_unused_dependencies(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for style in ('feminine', 'neutral', 'athletic'):
                with self.subTest(style=style):
                    rules = copy.deepcopy(load_rules(RULES))
                    rules['promptModules']['style'] = style
                    rule_file = root / (style + '.json')
                    rule_file.write_text(json.dumps(rules), encoding='utf-8')
                    common = ['--scope', 'locomotion', '--rules', str(rule_file)]
                    for motion in MOTIONS:
                        common += ['--motion', motion]
                    cli = [sys.executable, str(ROOT / 'tools/produce_als_resources.py')]
                    output = root / style
                    subprocess.run(cli + ['plan', *common, '--output', str(output)], cwd=ROOT,
                                   check=True, capture_output=True, text=True)
                    inspection = json.loads(subprocess.run(cli + ['inspect', *common], cwd=ROOT,
                        check=True, capture_output=True, text=True).stdout)
                    graph = read_json(output / 'graph.json')
                    selected = inspection['selectedOrder']
                    self.assertEqual(inspection['externalInputs'], [])
                    deliveries = [graph['nodes'][key] for key in selected if graph['nodes'][key]['op'] == 'deliver']
                    self.assertEqual(len(deliveries), 6)
                    self.assertEqual({node['params']['contract']['semanticName'] for node in deliveries}, set(MOTIONS))
                    self.assertFalse(any('Stand' in key or 'Run' in key or 'discovery' in key for key in selected))
                    for node in deliveries:
                        self.assertEqual(node['params']['contract']['targetGaitPolicy'], 'als-forward-v1')
                    for motion, speed, direction in zip(MOTIONS, (6, 5, 1.5), ([0, 1], [0, 1], [0, -1])):
                        params = graph['nodes']['candidate:' + motion]['params']
                        self.assertEqual(params['seconds'], 3)
                        self.assertEqual(params['rootPath']['velocity'], speed)
                        self.assertEqual(params['rootPath']['direction'], direction)
                        self.assertTrue(params['preserveReferencePelvisPath'])
                        self.assertTrue(params['nativePhaseEvidence'])
                        self.assertEqual(params['poseConstraintMode'], 'feet_only')
                        self.assertEqual(params['supportAnchorCount'], 2)
                        self.assertEqual(params['fullBodyPhaseFractions'], [.5])
                        self.assertEqual(params['fullBodyPhaseCycles'], 2)
                        repair = graph['nodes']['repair:' + motion + ':generated']['params']
                        self.assertEqual(repair['repairSelection'], 'bounded')
                        self.assertEqual(repair['maximumRepairAttempts'], 24)


if __name__ == '__main__':
    unittest.main()

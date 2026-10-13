"""Style changes must not silently change direction, gait phase or cache identity."""
import copy
import json
import unittest
from pathlib import Path

from .prompt_modules import build_ablation_plan, compile_prompt, resolve_prompt


def example():
    return json.loads(Path(__file__).with_name('prompt_module_examples.json').read_text(encoding='utf-8'))['promptModules']


class PromptModuleTests(unittest.TestCase):
    def test_three_styles_keep_functional_intent_and_crouch_excludes_upright(self):
        config = example()
        claims = []
        for style in ('neutral', 'feminine', 'athletic'):
            result = compile_prompt(config, 'Crouch_Walk_Backward', style=style)
            self.assertTrue(result['prompt'].startswith('A person is walking backward'))
            self.assertNotIn('upright', result['prompt'])
            claims.append({key: result['receipt']['claims'][key] for key in ('action', 'direction', 'phase', 'stance')})
        self.assertEqual(claims[0], claims[1])
        self.assertEqual(claims[1], claims[2])
        feminine = compile_prompt(config, 'Run_Forward')['prompt']
        self.assertIn('upper arms held steady', feminine)
        self.assertIn('forearms swinging', feminine)
        athletic = compile_prompt(config, 'Run_Forward', style='athletic')['prompt']
        self.assertIn('coordinated upper-arm', athletic)
        self.assertNotIn('upper arms held steady', athletic)

    def test_legacy_precedence_preserves_exact_bytes(self):
        rules = {'motionPromptOverrides': {'Walk_Forward': ' override  '}}
        self.assertEqual(resolve_prompt(rules, 'Walk_Forward', 'fallback')['prompt'], ' override  ')
        self.assertEqual(resolve_prompt(rules, 'Walk_Forward', 'fallback', explicit_prompt=' explicit  '),
                         {'prompt': ' explicit  ', 'receipt': None})
        rules['promptModules'] = example()
        self.assertEqual(resolve_prompt(rules, 'Jump_Loop', ' unchanged\n'), {'prompt': ' unchanged\n', 'receipt': None})
        self.assertNotEqual(resolve_prompt(rules, 'Walk_Forward', 'fallback', explicit_prompt='old')['prompt'], 'old')

    def test_one_module_removal_does_not_remove_forearm_intent(self):
        config = example()
        result = compile_prompt(config, 'Sprint', remove_modules=['steady-upper-arms'])
        self.assertNotIn('upper arms held steady', result['prompt'])
        self.assertIn('forearms swinging', result['prompt'])
        self.assertNotIn('upperArmMotion', result['receipt']['claims'])
        self.assertEqual(result['receipt']['claims']['handMotion'], 'visible')

    def test_conflicting_upper_arm_intent_is_rejected_before_generation(self):
        with self.assertRaisesRegex(ValueError, 'Conflicting claim upperArmMotion'):
            compile_prompt(example(), 'Sprint', add_modules=['athletic-arms'])

    def test_style_cannot_replace_functional_modules_or_claims(self):
        config = example()
        config['styles']['feminine'].append('backward')
        with self.assertRaisesRegex(ValueError, 'Style profiles cannot replace'):
            compile_prompt(config, 'Sprint')
        config = example()
        config['modules']['feminine-quality']['claims']['phase'] = 'accelerating'
        with self.assertRaisesRegex(ValueError, 'cannot override functional'):
            compile_prompt(config, 'Sprint')

    def test_numerical_constraint_intent_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'constraint claim direction'):
            compile_prompt(example(), 'Walk_Backward', expected_claims={'direction': 'forward'})
        with self.assertRaisesRegex(ValueError, 'constraint claim phase'):
            compile_prompt(example(), 'Sprint_Acceleration', expected_claims={'phase': 'accelerating'})

    def test_hash_is_order_independent_but_sensitive_to_effective_prompt(self):
        config = example()
        first = compile_prompt(config, 'Walk_Forward')
        shuffled = copy.deepcopy(config)
        shuffled['motions']['Walk_Forward']['modules'].reverse()
        shuffled['styles']['feminine'].reverse()
        shuffled['modules']['athletic-quality']['text'] = 'another unused description'
        self.assertEqual(first['receipt']['sha256'], compile_prompt(shuffled, 'Walk_Forward')['receipt']['sha256'])
        shuffled['modules']['moving-forearms']['text'] = 'forearms swinging with large visible hand travel'
        self.assertNotEqual(first['receipt']['sha256'], compile_prompt(shuffled, 'Walk_Forward')['receipt']['sha256'])

    def test_missing_modules_inheritance_cycles_and_typo_removals_fail(self):
        config = example()
        config['motions']['Walk_Forward'] = {'base': 'Sprint'}
        with self.assertRaisesRegex(ValueError, 'inheritance cycle'):
            compile_prompt(config, 'Sprint')
        with self.assertRaisesRegex(ValueError, 'removes absent'):
            compile_prompt(example(), 'Sprint', remove_modules=['typo'])
        with self.assertRaisesRegex(ValueError, 'Unknown prompt module'):
            compile_prompt(example(), 'Sprint', add_modules=['unknown'])

    def test_requires_claim_prevents_accidental_omission(self):
        config = example()
        config['modules']['moving-forearms']['requires'] = {'upperArmMotion': 'steady'}
        with self.assertRaisesRegex(ValueError, 'requires upperArmMotion'):
            compile_prompt(config, 'Sprint', remove_modules=['steady-upper-arms'])

    def test_ablation_plan_pairs_seeds_and_preserves_all_generation_controls(self):
        controls = dict(model='Kimodo-SOMA-RP-v1.1', skeleton='SOMA77', seconds=4, fps=30,
                        steps=100, cfgWeights=[2., 2.], constraintFingerprint='a' * 64,
                        decoder='fixed-decoder-version')
        plan = build_ablation_plan(example(), 'Sprint', generation=controls, seeds=[11, 23],
                                   styles=['neutral', 'athletic'], remove_modules=['steady-upper-arms'])
        self.assertEqual(plan['status'], 'planned')
        self.assertEqual(len(plan['experiments']), 8)
        for row in plan['experiments']:
            self.assertEqual(row['generation'], controls)
            self.assertEqual(row['promptReceipt']['claims']['action'], 'sprint')
            self.assertEqual(row['status'], 'planned')
        plan['experiments'][0]['generation']['cfgWeights'][0] = 0.
        self.assertEqual(controls['cfgWeights'][0], 2.)
        self.assertEqual(plan['experiments'][1]['generation']['cfgWeights'][0], 2.)

    def test_ablation_rejects_functional_or_inactive_removal_and_incomplete_controls(self):
        controls = dict(model='m', skeleton='s', seconds=4, fps=30, steps=100,
                        cfgWeights=[2., 2.], constraintFingerprint=None)
        with self.assertRaisesRegex(ValueError, 'constraint claim direction'):
            build_ablation_plan(example(), 'Sprint', generation=controls, seeds=[1], remove_modules=['forward'])
        with self.assertRaisesRegex(ValueError, 'inactive module'):
            build_ablation_plan(example(), 'Sprint', generation=controls, seeds=[1], remove_modules=['inclined-crouch'])
        del controls['constraintFingerprint']
        with self.assertRaisesRegex(ValueError, 'requires'):
            build_ablation_plan(example(), 'Sprint', generation=controls, seeds=[1])

    def test_malformed_module_data_is_not_silently_ignored(self):
        for mutation in ('when', 'claims', 'slot', 'text'):
            config = example()
            config['modules']['moving-forearms'][mutation] = {'wrong': 1} if mutation in ('when', 'claims') else None
            with self.assertRaises(ValueError):
                compile_prompt(config, 'Sprint')
        config = example()
        config['modules']['moving-forearms']['text'] = 'forearms swinging {distance}'
        with self.assertRaisesRegex(ValueError, 'materialized'):
            compile_prompt(config, 'Sprint')

    def test_ablation_rejects_unbound_or_invalid_generation_controls(self):
        controls = dict(model='m', skeleton='s', seconds=4, fps=30, steps=100,
                        cfgWeights=[2., 2.], constraintFingerprint=None)
        for key, value in (('constraintFingerprint', 'placeholder'), ('seconds', 11),
                           ('cfgWeights', [float('nan'), 2.]), ('fps', True), ('model', '')):
            with self.subTest(key=key):
                with self.assertRaises(ValueError):
                    build_ablation_plan(example(), 'Sprint', generation=dict(controls, **{key: value}), seeds=[1])


if __name__ == '__main__':
    unittest.main()

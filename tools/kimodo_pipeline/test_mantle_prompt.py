"""Prompt configuration must affect real DAG inputs without changing motion contracts."""
import copy
import hashlib
import json
import unittest

from .graph import build_graph, load_rules, read_json
from .mantle_prompt import CATALOG, resolve_prompt
from .paths import PIPELINE_ROOT, contract_path

PROFILES = {
    'Mantle_Low_Left': {'hands': ('right',)},
    'Mantle_Low_Right': {'hands': ('left',)},
    'Mantle_High': {'hands': ('left', 'right')},
}


class MantlePromptTests(unittest.TestCase):
    def test_default_preserves_previous_text_for_every_support_side(self):
        for name, profile in PROFILES.items():
            with self.subTest(name=name):
                hands = 'both hands' if len(profile['hands']) == 2 else profile['hands'][0] + ' hand'
                expected = ('A person mantles onto a ' + ('high' if 'High' in name else 'low') +
                    ' ledge, planting ' + hands + ' on the top surface, palms down and fingers forward. '
                    'Keep the supporting palms fixed while pulling and pushing the body up, '
                    'then release the hands and finish standing with both feet on the ledge.')
                prompt, receipt = resolve_prompt(name, profile, {})
                self.assertEqual(prompt, expected)
                self.assertEqual(receipt['semanticContract']['supportHands'], list(profile['hands']))

    def test_optional_style_and_modifiers_can_be_replaced_or_removed(self):
        profile = PROFILES['Mantle_Low_Left']
        base, base_receipt = resolve_prompt('Mantle_Low_Left', profile, {})
        rules = {'mantlePrompt': {'recipe': 'cautious', 'modifiers': [
            {'id': 'relaxed-shoulders', 'text': 'The shoulders remain relaxed.'}]}}
        styled, receipt = resolve_prompt('Mantle_Low_Left', profile, rules)
        self.assertIn('cautiously', styled)
        self.assertTrue(styled.endswith('The shoulders remain relaxed.'))
        self.assertEqual(receipt['semanticContract'], base_receipt['semanticContract'])
        rules['mantlePrompt'].update(style=None, modifiers=[])
        self.assertEqual(resolve_prompt('Mantle_Low_Left', profile, rules)[0], base)
        rules['mantlePrompt'].update(style='custom', styles={'custom': {'text': 'The person rises calmly.'}})
        self.assertTrue(resolve_prompt('Mantle_Low_Left', profile, rules)[0].endswith('rises calmly.'))

    def test_required_modules_cannot_be_deleted_but_text_is_replaceable(self):
        profile = PROFILES['Mantle_High']
        for replacement in (None, '', {'text': ''}):
            with self.subTest(replacement=replacement), self.assertRaisesRegex(ValueError, 'module|expected'):
                resolve_prompt('Mantle_High', profile, {'mantlePrompt': {'modules': {'contact': replacement}}})
        text, _ = resolve_prompt('Mantle_High', profile, {'mantlePrompt': {'modules': {
            'action': 'A person climbs onto a {ledge} ledge,',
            'contact': 'using {hands} to support the rise.'}}})
        self.assertIn('using both hands', text)

    def test_declared_incompatible_motion_contract_is_rejected(self):
        incompatible = ({'action': 'jump'}, {'repetitions': 2}, {'repetitions': True},
            {'supportHands': []}, {'supportHands': ['left']}, {'handContactRequired': False},
            {'finalSupport': 'one-foot'}, {'unknownSemantic': 'ignored'})
        for requirement in incompatible:
            with self.subTest(requirement=requirement), self.assertRaisesRegex(ValueError, 'contract|requirement'):
                resolve_prompt('Mantle_Low_Left', PROFILES['Mantle_Low_Left'], {
                    'mantlePrompt': {'modifiers': [{'id': 'declared-change', 'text': 'A new motion.',
                                                   'requires': requirement}]}})
        resolve_prompt('Mantle_Low_Left', PROFILES['Mantle_Low_Left'], {'mantlePrompt': {
            'requires': {'supportHands': ['right'], 'handContactRequired': True}}})

    def test_no_nlp_blacklist_is_claimed_as_semantic_verification(self):
        # Arbitrary override remains editable; geometry/source gates decide validity.
        override = 'A person jumps without using either hand.'
        text, receipt = resolve_prompt('Mantle_High', PROFILES['Mantle_High'], {
            'motionPromptOverrides': {'Mantle_High': override}})
        self.assertEqual(text, override)
        self.assertFalse(receipt['freeTextSemanticsVerified'])
        self.assertTrue(receipt['semanticContract']['handContactRequired'])
        self.assertEqual(receipt['selection'], 'motionPromptOverrides')

    def test_duplicate_modifier_unknown_schema_and_bad_templates_fail_early(self):
        bad = [dict(schemaVersion=2), dict(schemaVersion=True), dict(recipe='missing'), dict(style='missing'),
               dict(modules={'typo': 'ignored'}), dict(styles=[]), dict(modifiers={}),
               dict(modifiers=[{'id': 'x', 'text': 'A.'}, {'id': 'x', 'text': 'B.'}]),
               dict(modules={'action': 'A person does {unknown}.'}),
               dict(modules={'action': 'A person does {hands.__class__}.'}),
               dict(modules={'action': 'A person does {{unresolved}}.'})]
        for config in bad:
            with self.subTest(config=config), self.assertRaises(ValueError):
                resolve_prompt('Mantle_High', PROFILES['Mantle_High'], {'mantlePrompt': config})

    def test_receipt_hash_tracks_prompt_and_preserves_input_objects(self):
        rules = {'mantlePrompt': {'recipe': 'cautious'}}
        before = copy.deepcopy(rules)
        prompt, receipt = resolve_prompt('Mantle_High', PROFILES['Mantle_High'], rules)
        self.assertEqual(rules, before)
        self.assertEqual(receipt, resolve_prompt('Mantle_High', PROFILES['Mantle_High'], rules)[1])
        self.assertEqual(receipt['resolvedPromptSha256'], hashlib.sha256(prompt.encode()).hexdigest())
        rules['mantlePrompt']['recipe'] = 'light'
        light = resolve_prompt('Mantle_High', PROFILES['Mantle_High'], rules)[1]
        self.assertNotEqual(receipt['receiptSha256'], light['receiptSha256'])
        self.assertEqual(receipt['semanticContract'], light['semanticContract'])

    def test_unused_catalog_style_does_not_invalidate_current_prompt_identity(self):
        catalog = json.loads(CATALOG.read_text(encoding='utf-8'))
        first = resolve_prompt('Mantle_High', PROFILES['Mantle_High'], {}, catalog=catalog)
        catalog['styles']['cautious']['text'] = 'A different unused style.'
        self.assertEqual(first, resolve_prompt('Mantle_High', PROFILES['Mantle_High'], {}, catalog=catalog))

    def test_real_graph_uses_override_and_records_actual_text(self):
        rules = load_rules(PIPELINE_ROOT / 'tools/kimodo_pipeline/mantle_prompt_cautious_rules.json')
        rules['motionPromptOverrides']['Mantle_Low_Left'] = 'A person climbs a low ledge using the right hand.'
        graph = build_graph(read_json(contract_path()), rules)
        node = graph['nodes']['candidate:Mantle_Low_Left']['params']
        self.assertEqual(node['prompt'], rules['motionPromptOverrides']['Mantle_Low_Left'])
        self.assertEqual(node['mantlePromptReceipt']['resolvedPrompt'], node['prompt'])
        self.assertEqual(node['mantlePromptReceipt']['selection'], 'motionPromptOverrides')
        self.assertIn('cautiously', graph['nodes']['candidate:Mantle_Low_Right']['params']['prompt'])

    def test_style_recipes_change_only_generation_text_and_receipt(self):
        contracts = read_json(contract_path())
        variants = [build_graph(contracts, load_rules(PIPELINE_ROOT / (
            'tools/kimodo_pipeline/mantle_prompt_' + recipe + '_rules.json')))
            for recipe in ('default', 'cautious', 'light')]
        for name in PROFILES:
            nodes = [copy.deepcopy(graph['nodes']['candidate:' + name]) for graph in variants]
            prompts = [node['params'].pop('prompt') for node in nodes]
            self.assertEqual(len(set(prompts)), 3)
            for node in nodes:
                node['params'].pop('mantlePromptReceipt')
            self.assertEqual(nodes[0], nodes[1])
            self.assertEqual(nodes[1], nodes[2])

    def test_constraint_policy_is_explicit_and_independent_of_style(self):
        rules = load_rules(PIPELINE_ROOT / 'tools/kimodo_pipeline/mantle_prompt_light_rules.json')
        rules['mantleConstraintPolicy'] = 'sparse-contact-v1'
        graph = build_graph(read_json(contract_path()), rules)
        self.assertEqual(graph['nodes']['candidate:Mantle_High']['params']['mantleConstraintPolicy'],
                         'sparse-contact-v1')
        rules['mantleConstraintPolicy'] = 'unrecognized-policy'
        with self.assertRaisesRegex(ValueError, 'constraint policy'):
            build_graph(read_json(contract_path()), rules)


if __name__ == '__main__':
    unittest.main()

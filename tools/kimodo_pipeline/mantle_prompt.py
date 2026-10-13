"""Compose auditable Mantle prompts without putting style into the motion contract.

Structured requirements detect declared contradictions. Arbitrary English is not
semantically certified; the independent generated-motion gates remain necessary.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from string import Formatter

from .prompt_modules import fingerprint as _hash

POLICY = 'mantle-modular-prompt-v1'
CATALOG = Path(__file__).with_name('mantle_prompt_modules.json')
REQUIRED_MODULES = ('action', 'contact', 'transition')


def _requirements(requirements, contract, label):
    if not isinstance(requirements, dict):
        raise ValueError(label + ': requires must be an object')
    for key, value in requirements.items():
        if key not in contract:
            raise ValueError(label + ': unknown semantic requirement ' + key)
        if type(value) is not type(contract[key]) or value != contract[key]:
            raise ValueError(label + ': conflicts with Mantle contract: ' + key)


def _module(value, context, contract, label, *, optional=False):
    if isinstance(value, str):
        value = dict(text=value)
    if not isinstance(value, dict) or set(value) - {'text', 'requires', 'id'}:
        raise ValueError(label + ': expected text and optional requires')
    _requirements(value.get('requires', {}), contract, label)
    source = value.get('text')
    if not isinstance(source, str) or (not optional and not source.strip()):
        raise ValueError(label + ': required module text cannot be empty')
    try:
        fields = list(Formatter().parse(source))
        if any(field is not None and (field not in context or spec or conversion)
               for _, field, spec, conversion in fields):
            raise ValueError('Only ledge and hands placeholders are supported')
        text = source.format(**context).strip()
    except (KeyError, ValueError, IndexError) as exc:
        raise ValueError(label + ': invalid prompt template: ' + str(exc)) from exc
    if '{' in text or '}' in text:
        raise ValueError(label + ': unresolved prompt template')
    return dict(id=label, text=text, requires=copy.deepcopy(value.get('requires', {})))


def resolve_prompt(name, profile, rules, *, catalog=None):
    """Return resolved text plus a JSON receipt included in the graph identity.

    motionPromptOverrides[semantic] has highest text priority. With no settings,
    output text is byte-for-byte identical to the former graph.py Mantle prompt.
    """
    catalog = json.loads(CATALOG.read_text(encoding='utf-8')) if catalog is None else copy.deepcopy(catalog)
    if type(catalog.get('schemaVersion')) is not int or catalog['schemaVersion'] != 1:
        raise ValueError('Unsupported Mantle prompt catalog schema')
    hands = list(profile['hands'])
    if not hands or len(set(hands)) != len(hands) or set(hands) - {'left', 'right'}:
        raise ValueError('Mantle prompt requires explicit valid support hands')
    contract = dict(action='mantle', repetitions=1, supportHands=hands,
                    handContactRequired=True, finalSupport='both-feet')
    context = dict(ledge='high' if 'High' in name else 'low',
                   hands='both hands' if len(hands) == 2 else hands[0] + ' hand')
    config = rules.get('mantlePrompt', {})
    if not isinstance(config, dict) or set(config) - {
            'schemaVersion', 'recipe', 'modules', 'style', 'styles', 'modifiers', 'requires'}:
        raise ValueError('Unknown Mantle prompt configuration field')
    if type(config.get('schemaVersion', 1)) is not int or config.get('schemaVersion', 1) != 1:
        raise ValueError('Unsupported Mantle prompt configuration schema')
    _requirements(config.get('requires', {}), contract, 'mantlePrompt')
    recipe_name = config.get('recipe', 'default')
    if not isinstance(recipe_name, str) or recipe_name not in catalog['recipes']:
        raise ValueError('Unknown Mantle prompt recipe: ' + str(recipe_name))
    recipe = catalog['recipes'][recipe_name]
    module_overrides = config.get('modules', {})
    if not isinstance(module_overrides, dict) or set(module_overrides) - set(REQUIRED_MODULES):
        raise ValueError('Mantle modules must be action, contact or transition')
    modules = [_module(module_overrides.get(key, catalog['modules'][key]), context,
                       contract, key) for key in REQUIRED_MODULES]
    styles = dict(catalog['styles'])
    custom_styles = config.get('styles', {})
    if not isinstance(custom_styles, dict):
        raise ValueError('Mantle styles must be an object')
    styles.update(custom_styles)
    style_name = config.get('style', recipe.get('style'))
    if style_name is not None:
        if not isinstance(style_name, str) or style_name not in styles:
            raise ValueError('Unknown Mantle prompt style: ' + str(style_name))
        modules.append(_module(styles[style_name], context, contract,
                               'style:' + style_name, optional=True))
    modifiers = config.get('modifiers', recipe.get('modifiers', []))
    if not isinstance(modifiers, list):
        raise ValueError('Mantle modifiers must be an array; use [] to remove them')
    modifier_ids = set()
    for modifier in modifiers:
        if not isinstance(modifier, dict) or not isinstance(modifier.get('id'), str) or not modifier['id'].strip():
            raise ValueError('Each Mantle modifier needs a nonempty id')
        if modifier['id'] in modifier_ids:
            raise ValueError('Duplicate Mantle modifier id: ' + modifier['id'])
        modifier_ids.add(modifier['id'])
        modules.append(_module(modifier, context, contract, 'modifier:' + modifier['id']))
    composed = ' '.join(module['text'] for module in modules if module['text'])
    overrides = rules.get('motionPromptOverrides', {})
    override = overrides.get(name)
    if name in overrides:
        if not isinstance(override, str) or not override.strip() or '{' in override or '}' in override:
            raise ValueError('Mantle motionPromptOverrides must be nonempty resolved text')
        prompt = override
        selection = 'motionPromptOverrides'
    else:
        prompt = composed
        selection = 'modules'
    receipt = dict(schemaVersion=1, policy=POLICY, semanticName=name,
        recipe=recipe_name, style=style_name, selection=selection,
        semanticContract=contract, modules=modules, composedPrompt=composed,
        resolvedPrompt=prompt, resolvedPromptSha256=hashlib.sha256(prompt.encode()).hexdigest(),
        structuredRequirementsChecked=True,
        freeTextSemanticsVerified=False,
        note='Prompt intent is not a motion-quality certificate; keep source, target and runtime gates.')
    receipt['receiptSha256'] = _hash(receipt)
    return prompt, receipt

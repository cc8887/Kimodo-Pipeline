"""Deterministic, opt-in motion prompts with explicit functional/style ownership.

This validates declared intent, not the generated motion or arbitrary English.
Numerical contact, trajectory and delivery gates remain independent.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import re


POLICY = 'motion-prompt-modules-v1'
SLOTS = ('action', 'direction', 'phase', 'stance', 'posture', 'arms', 'style')
FUNCTIONAL_SLOTS = frozenset(SLOTS[:4])
FUNCTIONAL_CLAIMS = frozenset(('action', 'direction', 'heading', 'phase', 'stance', 'contact'))


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()


def _mapping(value, label):
    if not isinstance(value, dict):
        raise ValueError(label + ' must be an object')
    return value


def _names(value, label):
    if not isinstance(value, (list, tuple)) or any(not isinstance(x, str) or not x for x in value):
        raise ValueError(label + ' must be a list of module names')
    if len(set(value)) != len(value):
        raise ValueError(label + ' contains duplicate module names')
    return list(value)


def _claims(value, label):
    result = _mapping(value, label)
    if any(not isinstance(key, str) or not key or
           not isinstance(item, (str, bool)) or item == '' for key, item in result.items()):
        raise ValueError(label + ' values must be nonempty strings or booleans')
    return result


def _merge_claims(result, source, owner):
    for key, value in source.items():
        if key in result and (type(result[key]) is not type(value) or result[key] != value):
            raise ValueError(f'Conflicting claim {key}: {result[key]!r} versus {value!r} ({owner})')
        result[key] = value


def _patch(names, remove, add, label):
    remove, add = _names(remove, label + '.remove'), _names(add, label + '.add')
    unknown = set(remove) - set(names)
    if unknown:
        raise ValueError(f'{label} removes absent modules: {sorted(unknown)}')
    result = [name for name in names if name not in remove] + add
    return _names(result, label)


def _recipe(config, semantic, visiting=()):
    motions = _mapping(config.get('motions'), 'promptModules.motions')
    if semantic not in motions:
        raise ValueError('No modular prompt recipe for ' + semantic)
    if semantic in visiting:
        raise ValueError('Prompt recipe inheritance cycle: ' + ' -> '.join((*visiting, semantic)))
    recipe = _mapping(motions[semantic], 'recipe ' + semantic)
    unknown = set(recipe) - {'base', 'modules', 'remove', 'add', 'style'}
    if unknown:
        raise ValueError('Unknown prompt recipe keys: ' + str(sorted(unknown)))
    if 'base' in recipe:
        if 'modules' in recipe:
            raise ValueError('A prompt recipe cannot specify both base and modules')
        names, style = _recipe(config, recipe['base'], (*visiting, semantic))
    else:
        names = _names(recipe.get('modules', []), semantic + '.modules')
        style = config.get('style', 'neutral')
    names = _patch(names, recipe.get('remove', []), recipe.get('add', []), semantic)
    return names, recipe.get('style', style)


def compile_prompt(config, semantic, *, style=None, add_modules=(), remove_modules=(),
                   expected_claims=None):
    """Compile one recipe; optional expected_claims come from numerical constraint intent.

    Returns {prompt, receipt}. The receipt is JSON data suitable for DAG params and
    source receipts; fingerprint covers only effective modules, not unused catalog entries.
    """
    _mapping(config, 'promptModules')
    if config.get('schemaVersion') != 1 or isinstance(config.get('schemaVersion'), bool):
        raise ValueError('Unsupported promptModules.schemaVersion')
    unknown = set(config) - {'schemaVersion', 'subject', 'modules', 'styles', 'style', 'motions'}
    if unknown:
        raise ValueError('Unknown promptModules keys: ' + str(sorted(unknown)))
    names, selected_style = _recipe(config, semantic)
    selected_style = selected_style if style is None else style
    styles = _mapping(config.get('styles', {'neutral': []}), 'promptModules.styles')
    if selected_style not in styles:
        raise ValueError('Unknown prompt style: ' + str(selected_style))
    style_names = _names(styles[selected_style], 'style ' + selected_style)
    names = _patch(names + style_names, remove_modules, add_modules, 'prompt selection')
    catalog = _mapping(config.get('modules'), 'promptModules.modules')
    modules = []
    for name in names:
        if name not in catalog:
            raise ValueError('Unknown prompt module: ' + name)
        module = copy.deepcopy(_mapping(catalog[name], 'module ' + name))
        if set(module) - {'slot', 'text', 'claims', 'when', 'requires'}:
            raise ValueError('Unknown module fields: ' + name)
        slot = module.get('slot')
        if slot not in SLOTS:
            raise ValueError('Unknown prompt slot: ' + str(slot))
        if name in style_names and slot in FUNCTIONAL_SLOTS:
            raise ValueError('Style profiles cannot replace action/direction/phase/stance: ' + name)
        raw_text = module.get('text')
        if not isinstance(raw_text, str) or not raw_text.strip() or re.search(r'[{}\r\n]', raw_text):
            raise ValueError('Module text must be a materialized single-line clause: ' + name)
        module['text'] = ' '.join(raw_text.strip().rstrip('.;,').split())
        if not module['text']:
            raise ValueError('Empty prompt clause: ' + name)
        module['claims'] = _claims(module.get('claims', {}), name + '.claims')
        if slot not in FUNCTIONAL_SLOTS and FUNCTIONAL_CLAIMS.intersection(module['claims']):
            raise ValueError('Style/body modules cannot override functional claims: ' + name)
        module['when'] = _claims(module.get('when', {}), name + '.when')
        module['requires'] = _claims(module.get('requires', {}), name + '.requires')
        if set(module['when']) - FUNCTIONAL_CLAIMS or slot in FUNCTIONAL_SLOTS and module['when']:
            raise ValueError('Conditions may only select style/body modules using functional claims')
        modules.append(dict(module, id=name))
    functional = {}
    for module in modules:
        if module['slot'] in FUNCTIONAL_SLOTS:
            _merge_claims(functional, module['claims'], module['id'])
    selected, omitted = [], []
    for module in modules:
        if all(key in functional and functional[key] == value for key, value in module['when'].items()):
            selected.append(module)
        else:
            omitted.append(module['id'])
    selected.sort(key=lambda module: (SLOTS.index(module['slot']), module['id']))
    for slot in FUNCTIONAL_SLOTS:
        count = sum(module['slot'] == slot for module in selected)
        if count > 1 or slot == 'action' and count != 1:
            raise ValueError(f'Expected {"one" if slot == "action" else "at most one"} {slot} module')
    claims = {}
    for module in selected:
        _merge_claims(claims, module['claims'], module['id'])
    for module in selected:
        for key, value in module['requires'].items():
            if key not in claims or claims[key] != value:
                raise ValueError(f'Module {module["id"]} requires {key}={value!r}')
    expected = _claims(expected_claims or {}, 'expected_claims')
    for key, value in expected.items():
        if key not in claims or claims[key] != value:
            raise ValueError(f'Prompt contradicts or omits constraint claim {key}={value!r}')
    subject = config.get('subject', 'A person')
    if not isinstance(subject, str) or not subject.startswith('A ') or re.search(r'[.{}\r\n]', subject):
        raise ValueError('Prompt subject must be a single subject beginning with A')
    subject = ' '.join(subject.split())
    primary = ' '.join(module['text'] for module in selected if module['slot'] in FUNCTIONAL_SLOTS)
    detail = ', '.join(module['text'] for module in selected if module['slot'] not in FUNCTIONAL_SLOTS)
    prompt = subject + ' ' + primary + (', with ' + detail if detail else '') + '.'
    effective = dict(policy=POLICY, semantic=semantic, subject=subject, modules=selected, claims=claims,
                     expectedClaims=expected, prompt=prompt)
    receipt = dict(effective, style=selected_style, omittedModules=sorted(omitted),
                   sha256=fingerprint(effective))
    return dict(prompt=prompt, receipt=receipt)


def resolve_prompt(rules, semantic, fallback_prompt, *, explicit_prompt=None, expected_claims=None):
    """Legacy precedence remains byte-exact for semantics without an opted-in recipe."""
    config = rules.get('promptModules')
    if config is not None:
        _mapping(config, 'promptModules')
        motions = _mapping(config.get('motions'), 'promptModules.motions')
        if semantic in motions:
            return compile_prompt(config, semantic, expected_claims=expected_claims)
    prompt = explicit_prompt or rules.get('motionPromptOverrides', {}).get(semantic) or fallback_prompt
    return dict(prompt=prompt, receipt=None)


def build_ablation_plan(config, semantic, *, generation, seeds, styles=(), remove_modules=()):
    """Plan paired experiments only. No generation or quality success is implied.

    Every row fixes the entire generation dictionary and varies only a style selection
    or deletion of one named module; all variants use the same list of seeds.
    """
    generation = copy.deepcopy(_mapping(generation, 'generation'))
    required = {'model', 'skeleton', 'seconds', 'fps', 'steps', 'cfgWeights', 'constraintFingerprint'}
    if required - set(generation) or 'seed' in generation:
        raise ValueError('Ablation generation requires ' + ', '.join(sorted(required)) + '; seeds are separate')
    duration = generation['seconds']
    if isinstance(duration, bool) or not isinstance(duration, (int, float)) or not math.isfinite(duration) or not 0 < duration <= 10:
        raise ValueError('Each prompt segment must be positive and at most 10 seconds')
    if any(not isinstance(generation[key], str) or not generation[key].strip() for key in ('model', 'skeleton')):
        raise ValueError('Ablation model and skeleton must be explicit nonempty strings')
    if any(isinstance(generation[key], bool) or not isinstance(generation[key], int) or generation[key] <= 0
           for key in ('fps', 'steps')):
        raise ValueError('Ablation fps and steps must be positive integers')
    weights = generation['cfgWeights']
    if not isinstance(weights, (list, tuple)) or len(weights) != 2 or any(
            isinstance(weight, bool) or not isinstance(weight, (int, float)) or
            not math.isfinite(weight) or weight < 0 for weight in weights):
        raise ValueError('Ablation requires explicit finite text and constraint CFG weights')
    constraint_hash = generation['constraintFingerprint']
    if constraint_hash is not None and (not isinstance(constraint_hash, str) or
                                       not re.fullmatch('[0-9a-f]{64}', constraint_hash)):
        raise ValueError('constraintFingerprint must be a SHA-256 or None for unconstrained generation')
    seeds = list(seeds)
    if not seeds or len(set(seeds)) != len(seeds) or any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in seeds):
        raise ValueError('Ablation seeds must be distinct nonnegative integers')
    baseline = compile_prompt(config, semantic)
    baseline_claims = {key: value for key, value in baseline['receipt']['claims'].items() if key in FUNCTIONAL_CLAIMS}
    variants = [('baseline', 'baseline', baseline)]
    for style in _names(styles, 'ablation styles'):
        if style == baseline['receipt']['style']:
            raise ValueError('Baseline style is already included')
        variants.append(('style:' + style, 'style', compile_prompt(config, semantic, style=style,
                         expected_claims=baseline_claims)))
    for name in _names(remove_modules, 'ablation removals'):
        if name not in [module['id'] for module in baseline['receipt']['modules']]:
            raise ValueError('Cannot ablate an inactive module: ' + name)
        variants.append(('remove:' + name, 'module-removal', compile_prompt(config, semantic,
                         remove_modules=[name], expected_claims=baseline_claims)))
    rows = []
    for seed in seeds:
        for name, factor, compiled in variants:
            row = dict(variant=name, changedFactor=factor, seed=seed, generation=copy.deepcopy(generation),
                       prompt=compiled['prompt'], promptReceipt=compiled['receipt'], status='planned')
            row['sha256'] = fingerprint(row)
            rows.append(row)
    return dict(policy=POLICY, status='planned', semantic=semantic, experiments=rows,
                controlsSha256=fingerprint(generation), sha256=fingerprint(rows))

"""Explicit reuse of upstream receipts and independent static gait sampling."""
from __future__ import annotations

import copy
from collections import Counter
from pathlib import Path

import numpy as np

from .graph import digest, read_json, topological_order
from .runner import file_hash, write_json

POSE_POLICY = 'generated-gait-phase-pose-v1'
REUSE_POLICY = 'matching-generation-intent-v1'


def _intent_params(node):
    """Compare effective node inputs, not the whole rules/catalog document."""
    params = copy.deepcopy(node['params'])
    if node['op'] == 'generate':
        params.setdefault('postProcessing', True)
        params.setdefault('cfgWeights', None)
        params.setdefault('anchors', [])
        params.setdefault('endEffectors', [])
        modular = params.get('promptModules')
        if modular is not None:
            fields = ('policy', 'semantic', 'subject', 'modules', 'claims', 'expectedClaims', 'prompt')
            if not isinstance(modular, dict) or any(field not in modular for field in fields):
                raise ValueError('Incomplete effective modular prompt receipt')
            effective = {field: modular[field] for field in fields}
            if digest(effective) != modular.get('sha256') or effective['prompt'] != params.get('prompt'):
                raise ValueError('Modular prompt fingerprint does not bind the effective generation prompt')
            # Style names, omitted modules and unused catalog entries are
            # diagnostic metadata. The compiled effective modules are bound.
            params['promptModules'] = dict(policy=modular['policy'], sha256=modular['sha256'])
    return params


def _consumed_dependencies(node):
    """Generation reads only its selected constraint branch's motion inputs."""
    if node['op'] != 'generate':
        return sorted(set(node['deps']) - {'rig:soma'})
    params = node['params']
    for field in ('jumpReference', 'airReference', 'mantleReference', 'gaitReference'):
        if params.get(field):
            dependencies = [params[field]]
            if field == 'gaitReference' and params.get('postureIdleReference'):
                dependencies.append(params['postureIdleReference'])
            return sorted(set(dependencies))
    return sorted({row['source'] for kind in ('anchors', 'endEffectors') for row in params.get(kind, [])})


def _verified_receipt_node(receipt):
    original = receipt['inputs']['node']
    if digest(receipt['inputs']) != receipt['identity']:
        raise ValueError('Upstream receipt identity changed: ' + receipt['node'])
    if original['id'] != receipt['node'] or original['op'] != receipt['op']:
        raise ValueError('Upstream node identity differs: ' + receipt['node'])
    return original


def _matching_intent(node, receipt, nodes, incoming, *, root=True, visiting=()):
    original = _verified_receipt_node(receipt)
    # A receipt produced by a previous explicit reuse still carries its
    # executed source. Never replace its ancestry with today's same-name node.
    if original['op'] == 'reuse_motion':
        return _matching_intent(node, original['params']['originalReceipt'], nodes, incoming,
                                root=root, visiting=visiting)
    compatible_gait = root and {node['op'], original['op']} <= {'gait_cycle', 'mirror'}
    if node['op'] != original['op'] and not compatible_gait:
        return False
    if (node['op'] == original['op'] and _intent_params(node) != _intent_params(original)):
        return False
    dependencies = _consumed_dependencies(node)
    if dependencies != _consumed_dependencies(original):
        return False
    identity = (receipt['node'], receipt['identity'])
    if identity in visiting:
        raise ValueError('Cycle in upstream generation evidence')
    for dependency in dependencies:
        expected = receipt['inputs'].get('dependencies', {}).get(dependency)
        parent = incoming.get(dependency)
        if parent is None or dependency not in nodes or expected is None or parent['identity'] != expected:
            return False  # A dependency name alone is not proof of matching conditions.
        parent_hashes = {name: output['sha256'] for name, output in parent['outputs'].items()}
        if parent_hashes != receipt['inputs']['dependencyOutputs'].get(dependency):
            raise ValueError('Upstream dependency payload identity differs: ' + dependency)
        if not _matching_intent(nodes[dependency], parent, nodes, incoming,
                                root=False, visiting=(*visiting, identity)):
            return False
    return True


def configure_pose_samples(nodes, rules):
    for name, parent in rules.get('sampledGaitPoses', {}).items():
        key, candidate = 'motion:' + name, 'candidate:' + parent
        if key not in nodes or nodes[key]['op'] != 'freeze' or candidate not in nodes:
            raise ValueError('A sampled pose requires an existing gait pose and generated candidate: ' + name)
        generator = nodes[candidate]
        parent_node = nodes['motion:' + parent]
        period = generator['params'].get('targetPeriodSeconds') or parent_node['params'].get('seconds')
        path = generator['params'].get('rootPath')
        if generator['op'] != 'generate' or not period or not path or path['kind'] != 'linear':
            raise ValueError('A sampled pose requires a linear gait candidate with a known period: ' + name)
        nodes[key] = dict(id=key, op='sample_pose', deps=['rig:soma', candidate], params=dict(
            source=candidate, semantic=name, parentSemantic=parent, policy=POSE_POLICY,
            periodSeconds=period, phase=.25,
            samplingMode='conditioned-cycle' if generator['params'].get('targetPeriodSeconds') else 'loop-crop',
            count=nodes[key]['params']['count'], seconds=generator['params']['seconds'], rootPath=path))
    # The right reference uses the same sampled left body, with the existing mirror.
    if 'Run_Left_Pose' in rules.get('sampledGaitPoses', {}):
        key = 'motion:Run_Right_Pose'
        nodes[key] = dict(id=key, op='mirror', deps=['rig:soma', 'motion:Run_Left_Pose', 'motion:Stand_Pose'],
            params=dict(source='motion:Run_Left_Pose', base='motion:Stand_Pose', mask='all'))


def reuse_upstream(graph, paths, *, derived_only=False):
    """Reuse byte-verified motion only when its effective generation intent matches."""
    graph = copy.deepcopy(graph)
    requested_nodes = copy.deepcopy(graph['nodes'])
    for path in paths:
        path = Path(path).resolve()
        incoming = read_json(path)
        if isinstance(incoming, list):
            incoming = {row['receipt']['node']: row['receipt'] for row in incoming if row.get('passed') is True}
        for key, receipt in incoming.items():
            if key not in graph['nodes'] or receipt['op'] not in ('generate', 'gait_cycle', 'mirror'):
                continue
            node = graph['nodes'][key]
            if node['op'] != receipt['op'] and not {node['op'],receipt['op']} <= {'gait_cycle','mirror'}:
                continue
            original = _verified_receipt_node(receipt)
            if receipt['node'] != key:
                raise ValueError('Upstream node identity differs: ' + key)
            if receipt.get('boundInput'):
                continue  # A binding alone is not evidence of Kimodo generation.
            if not _matching_intent(requested_nodes[key], receipt, requested_nodes, incoming):
                continue
            outputs = receipt['outputs']
            for payload in outputs.values():
                if file_hash(payload['path']) != payload['sha256']:
                    raise ValueError('Changed upstream output: ' + key)
            if 'motion.npz' not in outputs or 'constraint-reference.json' in outputs:
                raise ValueError('Upstream source is not generated motion: ' + key)
            if receipt['op'] != 'generate':
                gate = outputs.get('gait-validation.json')
                if gate is None or read_json(gate['path']).get('pass') is not True:
                    continue
                if 'authoring-trajectory.json' not in outputs:
                    raise ValueError('Upstream gait is missing its trajectory: ' + key)
            rig_hash = receipt['inputs']['dependencyOutputs']['rig:soma']['rig.json']
            posture_idle=node['params'].get('postureValidationIdle')
            dependencies = _consumed_dependencies(requested_nodes[key])
            dependency_hashes = {}
            for dependency in dependencies:
                hashes = receipt['inputs']['dependencyOutputs'][dependency]
                used = {name: sha for name, sha in hashes.items() if name in ('motion.npz', 'rig.json')}
                dependency_hashes[dependency] = used or hashes
            graph['nodes'][key] = dict(id=key, op='reuse_motion',
                deps=list(dict.fromkeys(['rig:soma', *dependencies, *([posture_idle] if posture_idle else [])])), params=dict(
                receiptFile=str(path), receiptSha256=file_hash(path), originalReceipt=receipt,
                originalNode=node, rigSha256=rig_hash, reusePolicy=REUSE_POLICY,
                generationIntentSha256=digest(_intent_params(requested_nodes[key])),
                dependencyMotionHashes=dependency_hashes))
            for sample in graph['nodes'].values():
                if sample['op']=='sample_pose' and sample['params']['source']==key:
                    p=original['params'];sample['params'].update(seconds=p['seconds'],rootPath=p['rootPath'])
                    if p.get('targetPeriodSeconds'):
                        sample['params'].update(periodSeconds=p['targetPeriodSeconds'],samplingMode='conditioned-cycle')
    if derived_only:
        for key, node in graph['nodes'].items():
            if node['op'] == 'generate':
                graph['nodes'][key] = dict(id=key, op='external', deps=['rig:soma'], params=dict(
                    reason='No matching completed upstream Kimodo receipt; derived scope never runs diffusion'))
    graph['order'] = topological_order(graph['nodes'])
    graph.setdefault('summary',{})['operations']=dict(sorted(Counter(n['op'] for n in graph['nodes'].values()).items()))
    graph['pipeline'] = 'derived' if derived_only else graph.get('pipeline', 'non-prop')
    return graph


def handle_reuse(runner, node, folder):
    import shutil
    from .motion import load_motion
    from .gait import _array_hash, revalidate_authored_cycle
    p = node['params']
    if file_hash(p['receiptFile']) != p['receiptSha256']:
        raise ValueError('Upstream receipt changed')
    if runner.output_of('rig:soma')['rig.json']['sha256'] != p['rigSha256']:
        raise ValueError('Upstream rig differs from the selected SOMA rig')
    for dependency, expected in p.get('dependencyMotionHashes', {}).items():
        current = runner.output_of(dependency)
        if any(name not in current or current[name]['sha256'] != sha for name, sha in expected.items()):
            raise ValueError('Upstream generation dependency differs from current motion: ' + dependency)
    outputs = p['originalReceipt']['outputs']
    for payload in outputs.values():
        if file_hash(payload['path']) != payload['sha256']:
            raise ValueError('Upstream payload changed')
    motion = load_motion(outputs['motion.npz']['path'], runner.rig())
    if float(motion['fps']) != runner.graph['rules']['fps']:
        raise ValueError('Upstream motion FPS differs from the production time domain')
    if 'gait-validation.json' in outputs:
        gate = read_json(outputs['gait-validation.json']['path'])
        expected = _array_hash(motion['local_rot_mats'], motion['root_positions'], motion['fps'])
        if not gate.get('pass') or gate.get('outputMotionSha256') != expected:
            raise ValueError('Upstream gait receipt does not bind its motion')
        target = p['originalNode']['params'].get('targetCount')
        if target is not None and len(motion['root_positions']) != target:
            raise ValueError('Upstream gait time domain differs')
        current_validation = revalidate_authored_cycle(motion, runner.rig(), gate)
    for name in ('motion.npz', 'gait-validation.json', 'authoring-trajectory.json', 'upper-body-validation.json'):
        if name in outputs:
            shutil.copy2(outputs[name]['path'], folder / name)
    if 'gait-validation.json' in outputs:
        gate.update(policy=current_validation['policy'], policySha256=current_validation['policySha256'],
                    cycleAnchorSide=current_validation['cycleAnchorSide'],
                    reuseValidation=current_validation)
        write_json(folder / 'gait-validation.json', gate)
    current=p['originalNode']['params']
    if current.get('postureValidationIdle'):
        from .locomotion_style import audit_posture
        posture=audit_posture(motion,runner.motion(current['postureValidationIdle']),runner.rig(),current['postureProfile'])
        posture['idleNode']=current['postureValidationIdle']
        write_json(folder/'locomotion-posture-validation.json',posture)
        if not posture['passed']:
            raise ValueError('Reused locomotion does not satisfy the current posture profile')
    write_json(folder / 'reuse-provenance.json', p)


def sample_pose(source, rig, params):
    from .motion import freeze, remove_planar_path
    if params['policy'] != POSE_POLICY or not 0 <= params['phase'] < 1:
        raise ValueError('Invalid sampled pose policy/phase')
    period=round(params['periodSeconds']*float(source['fps']))
    if params['samplingMode']=='conditioned-cycle':
        frame=period+round(params['phase']*period)
    elif params['samplingMode']=='loop-crop':
        count=period+1
        start=max(0,min(round((len(source['root_positions'])-1)*.35),len(source['root_positions'])-count))
        frame=start+round(params['phase']*period)
    else:raise ValueError('Unknown pose sampling mode')
    in_place = remove_planar_path(source, params['rootPath'], params['seconds'], rig)
    result = freeze(in_place, frame, params['count'], rig)
    if not np.array_equal(result['local_rot_mats'][0], source['local_rot_mats'][frame]):
        raise ValueError('Pose extraction changed local body rotations')
    receipt = dict(policy=POSE_POLICY, passed=True, semanticName=params['semantic'],
        parentSemantic=params['parentSemantic'], sourceFrame=frame, sourceTimeSeconds=frame/float(source['fps']),
        sourceFrameCount=len(source['root_positions']), outputFrameCount=params['count'],
        removedPlanarTranslation=(source['root_positions'][frame]-in_place['root_positions'][frame]).tolist(),
        validationScope='static full pose only; no gait cycle acceptance', productionReady=False)
    return result, receipt


def retain_base_deliveries(pack_path, assets, receipts):
    from .godot_delivery import pack_root
    ROOT=pack_root(pack_path)
    pack=read_json(pack_path)
    for entry in pack['files']:
        if file_hash(ROOT/entry['path'].removeprefix('res://'))!=entry['sha256']:
            raise ValueError('Base pack payload changed: '+entry['path'])
    retarget=read_json(pack['nativeRetargetBundle'])
    author=Path(retarget['authoringBundle']);base=read_json(author)
    incoming=read_json(author.parent/'receipts.json')
    if incoming['rig:soma']['outputs']['rig.json']['sha256']!=receipts['rig:soma']['outputs']['rig.json']['sha256']:
        raise ValueError('Base and derived deliveries use different SOMA rigs')
    selected={a['node']:a for a in base['assets']}
    # Retained Jump/Air/Mantle deliveries can reference non-delivery dependency
    # receipts. Keep their executed provenance instead of dropping it at merge.
    combined=dict(incoming)
    retained_receipts={}
    for asset in base['assets']:
        for payload in asset['outputs'].values():
            if file_hash(payload['path'])!=payload['sha256']:raise ValueError('Base authoring output changed')
        # Node IDs may be overwritten by this batch. Retained motion must replay
        # its original dependency graph, rather than today's similarly named gait.
        origin=asset.get('sourceAuthoringReceipts') or dict(file=str(author.parent/'receipts.json'),
            sha256=file_hash(author.parent/'receipts.json'))
        if file_hash(origin['file'])!=origin['sha256']:
            raise ValueError('Retained source dependency evidence changed')
        original=read_json(origin['file']).get(asset['node'])
        if not original or original['identity']!=asset['identity'] or original['outputs']!=asset['outputs']:
            raise ValueError('Retained source does not bind its original authoring graph')
        retained_receipts[asset['node']]=original
        selected[asset['node']]=dict(asset,sourceAuthoringReceipts=origin)
    selected.update({a['node']:a for a in assets});combined.update(receipts)
    # Composed graphs can contain colliding node IDs from unselected branches.
    # Retained delivery entries must bind their certified origin, while incoming
    # replacements keep their own receipts and dependency graph.
    replaced={a['node'] for a in assets}
    combined.update({key:value for key,value in retained_receipts.items() if key not in replaced})
    return list(selected.values()),combined


def validate_pose_delivery(job):
    p=job.get('poseSamplingReceipt')
    if job['contract'].get('sampledPosePolicy')!=POSE_POLICY or not p:
        raise ValueError('Missing independent pose sampling evidence')
    for key,hash_key in (('file','sha256'),('sourcePoseMotionFile','sourcePoseMotionSha256')):
        if file_hash(p[key])!=p[hash_key]:raise ValueError('Pose sampling evidence changed')
    receipt=read_json(p['file'])
    source=receipt['sourceMotion']
    if receipt.get('passed') is not True or receipt['policy']!=POSE_POLICY or file_hash(source['path'])!=source['sha256']:
        raise ValueError('Invalid upstream pose sample')
    if file_hash(job['motionPath'])!=job['motionSha256'] or p['deliveredMotionSha256']!=job['motionSha256']:
        raise ValueError('Pose sampling does not bind the delivered motion')
    with np.load(job['motionPath'],allow_pickle=False) as data, np.load(p['sourcePoseMotionFile'],allow_pickle=False) as pose:
        for key in ('local_rot_mats','root_positions'):
            if not np.allclose(data[key],pose[key][0],atol=1e-12,rtol=0):
                raise ValueError('Static pose delivery changed the sampled body')


def targets(graph, contracts, pack):
    from .scope import is_prop
    missing = set(pack['missingNonPropSources'])
    return ['asset:' + row['edition'] + ':' + row['asset'] for row in contracts['assets']
            if not is_prop(row) and row['source'] in missing and
            graph['nodes']['motion:' + row['semanticName']]['op'] in ('sample_pose', 'freeze', 'offset', 'mirror', 'reuse_motion')]

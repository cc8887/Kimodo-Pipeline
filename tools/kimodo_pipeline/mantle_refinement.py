"""Receipt-backed Mantle refinement using Kimodo's public inference interface.

The Runner still owns cache identities, checkpoint hashes, authoring and delivery
gates. This adapter supplies already checked feedback conditions directly to the
SDK; it never patches a module or substitutes a reference clip for generation.
"""
from __future__ import annotations

import contextlib
import copy
import json
import os
from pathlib import Path
import shutil
import sys

from . import paths
from .batch import run_batch
from .graph import build_graph, digest, load_rules, read_json, validate_cfg_weights
from .runner import Runner, file_hash, write_json
from .mantle_constraint_feedback import build_feedback_constraints, ProjectionError

POLICY='mantle-refinement-v1'


class PhaseProgress:
    """Only compare like gates within the currently active refinement phase."""
    def __init__(self):
        self.phase=None;self.best=None;self.stalled=0

    def observe(self,phase,score):
        if phase not in ('source','target'):
            raise ValueError('Unknown refinement quality phase')
        if phase!=self.phase:
            self.phase=phase;self.best=None;self.stalled=0
        if score is not None:
            self.stalled=self.stalled+1 if self.best is not None and score>=self.best-1e-6 else 0
            self.best=score if self.best is None else min(self.best,score)
        return dict(phase=self.phase,bestNormalizedFailure=self.best,consecutiveNonImprovingRounds=self.stalled)


def checked_feedback(feedback):
    """Every experimental input is named in the generation-node identity."""
    required={'constraints','projectionReceipt','parentGeneratedMotion'}
    if not isinstance(feedback,dict) or set(feedback)!=required:
        raise ValueError('Feedback requires conditions, projection receipt and generated parent')
    for item in feedback.values():
        if not Path(item['file']).is_file() or file_hash(item['file'])!=item['sha256']:
            raise ValueError('Feedback input changed: '+str(item['file']))
    conditions=read_json(feedback['constraints']['file'])
    receipt=read_json(feedback['projectionReceipt']['file'])
    if not receipt.get('passed') or receipt.get('deliverable') is not False:
        raise ValueError('Only checked non-deliverable conditioning poses may refine generation')
    if receipt.get('constraintsSha256')!=digest(conditions):
        raise ValueError('Projection receipt does not bind the actual conditions')
    union={frame for condition in conditions for frame in condition['frame_indices']}
    if len(union)>=20 or any(len(condition['frame_indices'])>=20 for condition in conditions):
        raise ValueError('Feedback exceeds the sparse conditioning budget')
    return conditions,receipt


class MantleRefinementRunner(Runner):
    """Use the normal Runner except for an explicit materialized generation node."""
    def handle(self,node,folder,binding):
        params=node['params'];feedback=params.get('mantleFeedback')
        if node['op']!='generate' or feedback is None:
            return super().handle(node,folder,binding)
        if binding is not None:
            raise ValueError('Feedback generation cannot be replaced by a bound motion')
        conditions,receipt=checked_feedback(feedback)
        if not self.repo or not self.graph['rules'].get('reuseModel'):
            raise ValueError('Feedback SDK path requires the configured reusable Kimodo model')
        write_json(folder/'constraints.json',conditions)
        write_json(folder/'mantle-constraints.json',receipt)
        env=self.backend_env.copy()
        env.update(LOCAL_CACHE='True',HF_HUB_OFFLINE='1',TEXT_ENCODER_MODE='local',TEXT_ENCODER='llm2vec')
        env.setdefault('TEXT_ENCODER_DEVICE','cpu')
        env.pop('CHECKPOINT_DIR',None)
        os.environ.update(env);os.environ.pop('CHECKPOINT_DIR',None)
        sys.path.insert(0,str(self.repo))
        import torch
        from kimodo import load_model
        from kimodo.constraints import load_constraints_lst
        from kimodo.exports.motion_io import save_kimodo_npz
        from kimodo.tools import seed_everything
        from .motion import load_motion, save_motion
        cfg=validate_cfg_weights(params.get('cfgWeights'))
        with (folder/'generation.log').open('w',encoding='utf-8') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
            threads=self.graph['rules'].get('torchThreads')
            if threads is not None:
                if type(threads) is not int or threads<1:
                    raise ValueError('Invalid torchThreads')
                torch.set_num_threads(threads)
            if params['model'] not in self.models:
                device=self.graph['rules'].get('inferenceDevice','cuda:0' if torch.cuda.is_available() else 'cpu')
                if device not in ('cpu','cuda:0'):
                    raise ValueError('Unsupported inferenceDevice')
                frozen=None
                if self.graph['rules'].get('textEmbeddingCache'):
                    from .text_embeddings import FrozenTextEmbeddings
                    frozen=FrozenTextEmbeddings(self.graph['rules']['textEmbeddingCache'],self.checkpoint_identity(params['model']))
                print('KIMODO_MODEL_LOAD_BEGIN device='+device,flush=True)
                self.models[params['model']]=load_model(params['model'],device=device,default_family='Kimodo',text_encoder=frozen)
                print('KIMODO_MODEL_LOAD_OK',flush=True)
            model=self.models[params['model']]
            seed_everything(params['seed'])
            loaded=load_constraints_lst(str(folder/'constraints.json'),model.skeleton)
            guidance=dict(cfg_type='separated',cfg_weight=cfg) if cfg is not None else {}
            output=model([params['prompt']],[int(params['seconds']*params['fps'])],constraint_lst=loaded,
                num_denoising_steps=params['steps'],num_samples=1,multi_prompt=True,num_transition_frames=5,
                post_processing=params.get('postProcessing',True),return_numpy=True,**guidance)
            sample={key:(value[0] if hasattr(value,'shape') and len(value.shape)>0 and value.shape[0]==1 else value)
                    for key,value in output.items()}
            save_kimodo_npz(str(folder/'generated.npz'),sample)
        motion=load_motion(folder/'generated.npz',self.rig(),params['fps'])
        if len(motion['root_positions'])!=int(params['seconds']*params['fps']):
            raise ValueError('Kimodo frame count/FPS differs from the refinement plan')
        save_motion(folder/'motion.npz',motion,self.rig())
        write_json(folder/'generation-settings.json',params)


def verify_receipts(graph,receipts,authoring=None):
    """Resume only the exact completed nodes and payloads that were recorded."""
    if 'rules' in graph and digest(graph['rules'])!=graph.get('rulesDigest'):
        raise ValueError('Authoring graph rules digest changed')
    if authoring is not None and authoring.get('graphDigest')!=digest(graph):
        raise ValueError('Authoring batch does not bind this graph')
    for key,receipt in receipts.items():
        if (receipt['node']!=key or key not in graph['nodes'] or receipt['inputs']['node']!=graph['nodes'][key] or
                digest(receipt['inputs'])!=receipt['identity'] or
                receipt.get('op')!=graph['nodes'][key]['op'] or
                'rules' in receipt['inputs'] and receipt['inputs']['rules']!=graph.get('rulesDigest')):
            raise ValueError('Authoring graph no longer matches receipt: '+key)
        dependencies=graph['nodes'][key]['deps']
        if any(dependency not in receipts for dependency in dependencies):
            raise ValueError('Authoring dependency receipt missing: '+key)
        expected_identities={dependency:receipts[dependency]['identity'] for dependency in dependencies}
        expected_outputs={dependency:{name:item['sha256'] for name,item in receipts[dependency]['outputs'].items()}
                          for dependency in dependencies}
        if (receipt['inputs'].get('dependencies',{})!=expected_identities or
                receipt['inputs'].get('dependencyOutputs',{})!=expected_outputs):
            raise ValueError('Authoring dependency evidence differs: '+key)
        for payload in receipt['outputs'].values():
            if not Path(payload['path']).is_file() or file_hash(payload['path'])!=payload['sha256']:
                raise ValueError('Authoring payload changed: '+payload['path'])


def feedback_options(attempt,conditions):
    occupied={frame for condition in conditions for frame in condition['frame_indices']}
    remaining=19-len(occupied)
    return dict(maximum_added_frames=4 if attempt==1 else 2 if attempt==2 else max(1,min(3,remaining)),
                adaptive_margins=attempt>=3,reuse_existing_time_keys=remaining==0)


def author_failure_score(runner,semantic):
    """Worst normalized failing gate; preserve the full underlying gate report."""
    key='motion:'+semantic
    if key in runner.results:
        return 0.
    event=next((event for event in reversed(runner.events) if event.get('node')==key and event['status']=='failed'),None)
    if event is None or not event.get('evidence'):
        return None
    report=Path(event['evidence'])/'mantle-validation.json'
    if not report.is_file():
        return None
    failed=[float(check['value'])/float(check['limit']) for check in read_json(report)['checks']
            if not check['passed'] and check['limit']>0]
    return max(failed,default=0.)


def _complete(folder,graph,authoring,*,base_pack=None,runtime_cases=None):
    from .full_pipeline import complete_run
    runner=MantleRefinementRunner(graph,folder/'authoring',paths.configured_path('kimodo_repo'),paths.configured_path('python'))
    runner.results=read_json(runner.output/'receipts.json');verify_receipts(graph,runner.results,authoring)
    # A generated source may pass SOMA gates yet exceed target-proportion
    # adaptation limits. Do not manufacture a native/base-only success for it.
    semantics={node['params']['contract']['semanticName'] for key,node in graph['nodes'].items()
               if key in runner.results and node['op']=='deliver' and node['params']['contract']['family']=='mantle'}
    from .mantle_target_preflight import check_targets
    preflights=[check_targets(runner,semantic) for semantic in sorted(semantics)]
    origin=None
    if (runner.output/'full-pipeline').exists():
        # A previous partial/completed conversion is evidence, never proof of
        # newly requested options. Reuse only verified source receipts and run
        # fresh downstream stages in a separate continuation directory.
        origin=folder
        continuation_root=folder/'continuations';continuation_root.mkdir(exist_ok=True)
        index=1
        while True:
            continuation=continuation_root/('run-'+str(index).zfill(3))
            try:
                continuation.mkdir()
                break
            except FileExistsError:
                index+=1
        folder=continuation
        write_json(folder/'graph.json',graph)
        runner.output=folder/'authoring';runner.output.mkdir()
        write_json(runner.output/'receipts.json',runner.results)
        write_json(runner.output/'batch-result.json',authoring)
        write_json(folder/'resume-provenance.json',dict(originalAuthoringFolder=str(origin),
            originalReceiptsSha256=file_hash(origin/'authoring/receipts.json'),
            basePack=str(base_pack) if base_pack else None,runtimeCases=runtime_cases,
            reason='Fresh downstream verification from unchanged accepted source receipts'))
    if preflights:
        write_json(folder/'completion-target-preflight.json',preflights)
    if preflights and not any(report['acceptedTargets'] for report in preflights):
        evidence=runner.output/'full-pipeline';evidence.mkdir()
        result=dict(status='pipeline_complete_rejected',failedPhase='target-quality',authoring=authoring,
            targetPreflight=preflights,productionReady=False,activated=False,
            remainingStageStatus='blocked: no newly generated target passed the unchanged Mantle gates',
            evidence=str(evidence))
        write_json(evidence/'result.json',result)
        return result
    pack=paths.asset_root()/'assets/generated'/('als_nonprop_mantle_refined_'+digest(str(folder.resolve()))[:16])
    result=complete_run(runner,authoring,pack=pack,godot=paths.configured_path('godot'),dotnet=paths.configured_path('dotnet'),
        deps=paths.configured_path('offline_deps'),base_pack=base_pack,runtime_cases=runtime_cases)
    if origin is not None:
        result=dict(result,resumeContinuation=str(folder),originalAuthoringFolder=str(origin))
        write_json(folder/'continuation-result.json',result)
    return result


def refine(*,rules_path,semantic,output,max_feedback_rounds=8,base_pack=None,runtime_cases=None,device=None):
    """Normal entry always completes accepted sources through FBX/import/runtime."""
    if type(max_feedback_rounds) is not int or not 0<=max_feedback_rounds<=12:
        raise ValueError('Feedback round budget must be an integer between 0 and 12')
    output=Path(output).resolve()
    if output.exists():
        raise ValueError('Output already exists; preserve it and use --resume or a fresh path')
    output.mkdir(parents=True)
    rules=load_rules(rules_path);rules['reuseModel']=True
    rules.setdefault('mantleConstraintPolicy','sparse-contact-v1')
    if device is not None:
        rules['inferenceDevice']=device
    contracts=read_json(paths.contract_path())
    for asset in contracts['assets']:
        for evidence in asset['evidence']:
            if file_hash(paths.reference_root()/evidence['file'])!=evidence['sha256']:
                raise ValueError('Original ALS reference changed: '+evidence['file'])
    graph=build_graph(contracts,rules);graph['pipeline']='mantle'
    targets=[key for key,node in graph['nodes'].items() if node['op']=='deliver'
             and node['params']['contract']['family']=='mantle' and node['params']['contract']['semanticName']==semantic]
    if not targets or 'candidate:'+semantic not in graph['nodes']:
        raise ValueError('No configured Mantle generation for '+semantic)
    snapshot=output/'source-snapshot';snapshot.mkdir()
    for path in Path(__file__).parent.glob('*'):
        if path.is_file() and path.suffix in ('.py','.json') and not path.name.startswith('test_'):
            shutil.copyfile(path,snapshot/path.name)
    models={};checkpoints={};rounds=[];previous=None;previous_target=None;progress=PhaseProgress()
    for attempt in range(max_feedback_rounds+1):
        folder=output/('round-'+str(attempt));folder.mkdir()
        current=copy.deepcopy(graph)
        if previous is not None:
            params=current['nodes']['candidate:'+semantic]['params']
            parent_conditions=read_json(previous.output_of('candidate:'+semantic)['constraints.json']['path'])
            try:
                if previous_target is not None and not previous_target['passed']:
                    from .mantle_target_feedback import build_target_feedback_constraints
                    parent_key='motion:'+semantic
                    conditions,projection=build_target_feedback_constraints(params,previous.motion(parent_key),
                        previous.motion('reference:'+semantic+':native-mantle'),previous.rig(),parent_conditions)
                else:
                    parent_key='candidate:'+semantic
                    conditions,projection=build_feedback_constraints(params,previous.motion(parent_key),
                        previous.motion('reference:'+semantic+':native-mantle'),previous.rig(),parent_conditions,
                        **feedback_options(attempt,parent_conditions))
            except (ProjectionError,ValueError) as error:
                write_json(folder/'projection-failure.json',dict(error=str(error),receipt=getattr(error,'receipt',None)))
                rounds.append(dict(attempt=attempt,status='feedback_blocked',reason=str(error)))
                break
            condition_path=folder/'feedback-constraints.json';projection_path=folder/'feedback-projection.json'
            parent_motion=previous.output_of(parent_key)['motion.npz']
            projection['parentMotionNode']=parent_key
            projection['parentMotionSha256']=parent_motion['sha256']
            write_json(condition_path,conditions);write_json(projection_path,projection)
            params['mantleFeedback']=dict(constraints=dict(file=str(condition_path),sha256=file_hash(condition_path)),
                projectionReceipt=dict(file=str(projection_path),sha256=file_hash(projection_path)),
                parentGeneratedMotion=dict(file=parent_motion['path'],sha256=parent_motion['sha256']))
        write_json(folder/'graph.json',current)
        runner=MantleRefinementRunner(current,folder/'authoring',paths.configured_path('kimodo_repo'),paths.configured_path('python'))
        runner.models=models;runner.checkpoints=checkpoints
        authoring=run_batch(runner,targets)
        score=author_failure_score(runner,semantic)
        target_preflight=None
        if authoring['completedTargets']:
            from .mantle_target_preflight import check_targets,failure_score
            target_preflight=check_targets(runner,semantic)
            write_json(folder/'target-preflight.json',target_preflight)
            score=failure_score(target_preflight)
        phase_progress=progress.observe('target' if target_preflight is not None else 'source',score)
        rounds.append(dict(attempt=attempt,folder=str(folder),status=authoring['status'],completedTargets=authoring['completedTargets'],
            targetPreflightPassed=target_preflight['passed'] if target_preflight is not None else None,
            worstNormalizedFailure=score,phaseProgress=phase_progress))
        state=dict(policy=POLICY,semantic=semantic,rounds=rounds,latestAuthoringFolder=str(folder),
            productionReady=False,activated=False,status='pending_full_pipeline',basePack=str(base_pack) if base_pack else None)
        write_json(output/'refinement-state.json',state)
        previous=runner;previous_target=target_preflight
        if target_preflight is not None and target_preflight['passed'] or 'candidate:'+semantic not in runner.results or progress.stalled>=2:
            if progress.stalled>=2:
                state['stopReason']='Two feedback rounds did not improve the best measured '+progress.phase+' failure within this phase'
            break
    # Even source failure receives an explicit blocked complete_run report;
    # no native-only fallback can masquerade as successful generation.
    latest=Path(state['latestAuthoringFolder'])
    result=_complete(latest,read_json(latest/'graph.json'),read_json(latest/'authoring/batch-result.json'),
                     base_pack=base_pack,runtime_cases=runtime_cases)
    state.update(status=result['status'],pipeline=result,rounds=rounds)
    write_json(output/'refinement-state.json',state)
    return state


def resume(output,*,base_pack=None,runtime_cases=None):
    """Continue verified existing author receipts; generation is never rerun here."""
    output=Path(output).resolve()
    state_path=output/'refinement-state.json'
    state=read_json(state_path) if state_path.is_file() else None
    folder=Path(state['latestAuthoringFolder']) if state else output
    if not (folder/'graph.json').is_file() or not (folder/'authoring/receipts.json').is_file():
        raise ValueError('Resume requires an existing graph and authoring receipts')
    graph=read_json(folder/'graph.json');authoring=read_json(folder/'authoring/batch-result.json')
    result=_complete(folder,graph,authoring,base_pack=base_pack or (state or {}).get('basePack'),runtime_cases=runtime_cases)
    if state:
        state.update(status=result['status'],pipeline=result)
        write_json(state_path,state)
    else:
        write_json(folder/'resume-result.json',result)
    return result

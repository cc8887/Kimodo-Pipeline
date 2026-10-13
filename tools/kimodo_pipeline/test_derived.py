from .paths import project_root, asset_root, PIPELINE_ROOT
"""Static sampling and explicit source reuse must not certify failed gaits."""
import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from .derived import POSE_POLICY, sample_pose, reuse_upstream, retain_base_deliveries, validate_pose_delivery
from .graph import build_graph, digest, load_rules, read_json, topological_order
from .runner import Runner, write_json, file_hash
from .test_pipeline import rig_fixture, clip

ROOT=asset_root()


class DerivedTests(unittest.TestCase):
    def test_composed_receipt_collision_uses_verified_delivery_origin(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);rig=dict(outputs={'rig.json':dict(sha256='same-rig')})
            original=root/'original.json'
            write_json(original,{'existing':dict(identity='certified',outputs={})})
            origin=dict(file=str(original),sha256=file_hash(original))
            base=dict(node='existing',identity='certified',outputs={},sourceAuthoringReceipts=origin)
            write_json(root/'authoring.json',dict(assets=[base]))
            write_json(root/'receipts.json',{'rig:soma':rig,'existing':dict(identity='unselected',outputs={})})
            write_json(root/'retarget.json',dict(authoringBundle=str(root/'authoring.json')))
            pack=root/'assets/generated/als_nonprop_test/resource-pack.json';pack.parent.mkdir(parents=True)
            write_json(pack,dict(files=[],nativeRetargetBundle=str(root/'retarget.json')))
            assets,receipts=retain_base_deliveries(pack,[],{'rig:soma':rig,'existing':dict(identity='another-branch')})
            self.assertEqual(assets,[base])
            self.assertEqual(receipts['existing']['identity'],'certified')
            write_json(original,{'existing':dict(identity='tampered',outputs={})})
            with self.assertRaisesRegex(ValueError,'dependency evidence changed'):
                retain_base_deliveries(pack,[],{'rig:soma':rig})

    def reuse_fixture(self, root, *, modular=False):
        params = dict(semantic='Run', prompt='A person runs forward.', model='Kimodo-SOMA-RP-v1.1',
                      fps=30, seconds=3., seed=23, steps=100, cfgWeights=[2., 2.],
                      anchors=[], rootPath=dict(kind='linear', velocity=3.5, direction=[0, 1]),
                      postProcessing=True)
        if modular:
            from .prompt_modules import compile_prompt
            compiled = compile_prompt(dict(schemaVersion=1, styles={'neutral': []},
                modules={'run': dict(slot='action', text='runs forward', claims={'action': 'run'})},
                motions={'Run': dict(modules=['run'])}), 'Run')
            params.update(prompt=compiled['prompt'], promptModules=compiled['receipt'])
        graph = dict(nodes={'rig:soma': dict(id='rig:soma', op='rig', deps=[], params={}),
            'candidate': dict(id='candidate', op='generate', deps=['rig:soma'], params=params)},
            rules=dict(fps=30), rulesDigest='original-catalog')
        (root/'source.npz').write_bytes(b'content-addressed source fixture')
        inputs = dict(node=copy.deepcopy(graph['nodes']['candidate']),
                      dependencyOutputs={'rig:soma': {'rig.json': 'same-rig'}})
        receipt = dict(node='candidate', op='generate', inputs=inputs, identity=digest(inputs), boundInput=False,
                       outputs={'motion.npz': dict(path=str(root/'source.npz'), sha256=file_hash(root/'source.npz'))})
        path = root/'receipts.json'
        write_json(path, {'candidate': receipt})
        return graph, receipt, path

    def test_changed_generation_intent_never_reuses_old_motion(self):
        changes = dict(prompt='A person runs with graceful forearm motion.', model='other-model', fps=60,
                       seconds=4., seed=24, steps=50, cfgWeights=[1., 3.], postProcessing=False,
                       rootPath=dict(kind='linear', velocity=6., direction=[0, 1]),
                       anchors=[dict(source='pose', at=.5, fraction=.2)],
                       targetPeriodSeconds=.6, poseConstraintMode='lower_body',
                       supportAnchorCount=6, postureProfile={'bodyMotionGain': .2})
        with tempfile.TemporaryDirectory() as directory:
            graph, receipt, path = self.reuse_fixture(Path(directory))
            self.assertEqual('reuse_motion', reuse_upstream(graph, [path])['nodes']['candidate']['op'])
            for field, value in changes.items():
                with self.subTest(field=field):
                    changed = copy.deepcopy(graph)
                    changed['nodes']['candidate']['params'][field] = value
                    self.assertEqual('generate', reuse_upstream(changed, [path])['nodes']['candidate']['op'])
                    self.assertEqual('external', reuse_upstream(changed, [path], derived_only=True)['nodes']['candidate']['op'])
            # Explicit and implicit defaults express identical generation.
            del graph['nodes']['candidate']['params']['postProcessing']
            self.assertEqual('reuse_motion', reuse_upstream(graph, [path])['nodes']['candidate']['op'])

    def test_modular_prompt_reuse_ignores_unused_catalog_but_binds_effective_modules(self):
        with tempfile.TemporaryDirectory() as directory:
            graph, receipt, path = self.reuse_fixture(Path(directory), modular=True)
            graph['rulesDigest'] = 'unrelated-catalog-edit'
            graph['rules']['promptModules'] = {'unusedCatalogEntry': 'does not affect this prompt'}
            modular = graph['nodes']['candidate']['params']['promptModules']
            modular.update(style='renamed-profile-same-effective-modules', omittedModules=['unused-standing-style'])
            self.assertEqual('reuse_motion', reuse_upstream(graph, [path])['nodes']['candidate']['op'])
            modular['claims']['tone'] = 'calm'
            modular['sha256'] = digest({key: value for key, value in modular.items()
                                        if key not in ('sha256', 'style', 'omittedModules')})
            self.assertEqual('generate', reuse_upstream(graph, [path])['nodes']['candidate']['op'])
            modular['sha256'] = 'changed-without-binding-effective-modules'
            with self.assertRaisesRegex(ValueError, 'fingerprint'):
                reuse_upstream(graph, [path])

    def test_reusing_a_gait_cannot_hide_its_changed_generated_parent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            graph, parent, path = self.reuse_fixture(root)
            cycle = dict(id='cycle', op='gait_cycle', deps=['rig:soma', 'candidate'],
                         params=dict(source='candidate', targetCount=25))
            graph['nodes']['cycle'] = cycle
            write_json(root/'gate.json', {'pass': True})
            write_json(root/'trajectory.json', {'positions': []})
            hashes = {name: output['sha256'] for name, output in parent['outputs'].items()}
            inputs = dict(node=copy.deepcopy(cycle), dependencies={'candidate': parent['identity']},
                dependencyOutputs={'rig:soma': {'rig.json': 'same-rig'}, 'candidate': hashes})
            receipt = dict(node='cycle', op='gait_cycle', inputs=inputs, identity=digest(inputs), boundInput=False,
                outputs=dict(parent['outputs'], **{
                    'gait-validation.json': dict(path=str(root/'gate.json'), sha256=file_hash(root/'gate.json')),
                    'authoring-trajectory.json': dict(path=str(root/'trajectory.json'), sha256=file_hash(root/'trajectory.json'))}))
            write_json(path, {'candidate': parent, 'cycle': receipt})
            reused = reuse_upstream(graph, [path])
            self.assertEqual('reuse_motion', reused['nodes']['cycle']['op'])
            self.assertIn('candidate', reused['nodes']['cycle']['deps'])
            graph['nodes']['candidate']['params']['prompt'] = 'A person runs with a new upper-body style.'
            rejected = reuse_upstream(graph, [path])
            self.assertEqual('generate', rejected['nodes']['candidate']['op'])
            self.assertEqual('gait_cycle', rejected['nodes']['cycle']['op'])
            graph['nodes']['candidate'] = copy.deepcopy(parent['inputs']['node'])
            write_json(path, {'cycle': receipt})
            self.assertEqual('gait_cycle', reuse_upstream(graph, [path])['nodes']['cycle']['op'])

    def test_reuse_rejects_changed_constraint_dependency_motion_at_execution(self):
        from .derived import handle_reuse
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            graph, receipt, path = self.reuse_fixture(root)
            reused = reuse_upstream(graph, [path])['nodes']['candidate']
            reused['params']['dependencyMotionHashes'] = {'reference': {'motion.npz': 'old-motion'}}
            runner = SimpleNamespace(output_of=lambda key: {'rig.json': {'sha256': 'same-rig'}}
                if key == 'rig:soma' else {'motion.npz': {'sha256': 'changed-motion'}})
            with self.assertRaisesRegex(ValueError, 'generation dependency differs'):
                handle_reuse(runner, reused, root)

    def test_changed_anchor_intent_invalidates_only_its_actual_consumer(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            graph, receipt, path = self.reuse_fixture(root)
            reference = dict(id='reference', op='freeze', deps=['rig:soma'], params=dict(fraction=.2, count=1))
            graph['nodes']['reference'] = reference
            graph['nodes']['unused'] = dict(id='unused', op='generate', deps=['rig:soma'],
                                             params=dict(prompt='An unrelated action.'))
            candidate = graph['nodes']['candidate']
            candidate['deps'].extend(['reference', 'unused'])
            candidate['params']['anchors'] = [dict(source='reference', at=0., sourceFrame=0)]
            parent_inputs = dict(node=copy.deepcopy(reference), dependencyOutputs={'rig:soma': {'rig.json': 'same-rig'}})
            parent = dict(node='reference', op='freeze', inputs=parent_inputs, identity=digest(parent_inputs),
                          outputs=receipt['outputs'])
            receipt['inputs'].update(node=copy.deepcopy(candidate), dependencies={'reference': parent['identity']})
            receipt['inputs']['dependencyOutputs']['reference'] = {
                name: output['sha256'] for name, output in parent['outputs'].items()}
            receipt['identity'] = digest(receipt['inputs'])
            write_json(path, {'candidate': receipt, 'reference': parent})
            reused = reuse_upstream(graph, [path])
            self.assertEqual('reuse_motion', reused['nodes']['candidate']['op'])
            self.assertEqual(['rig:soma', 'reference'], reused['nodes']['candidate']['deps'])
            graph['nodes']['unused']['params']['prompt'] = 'Changed but still not a constraint input.'
            self.assertEqual('reuse_motion', reuse_upstream(graph, [path])['nodes']['candidate']['op'])
            graph['nodes']['reference']['params']['fraction'] = .7
            self.assertEqual('generate', reuse_upstream(graph, [path])['nodes']['candidate']['op'])

    def test_target_pose_gate_rejects_changed_body_even_with_updated_file_hash(self):
        from .motion import freeze, save_motion
        from .runner import file_hash
        rig=rig_fixture();source=clip(rig,90)
        params=dict(policy=POSE_POLICY,semantic='Lean_Pose',parentSemantic='Run_Forward',phase=.25,
            periodSeconds=.8,samplingMode='conditioned-cycle',count=1,seconds=3,
            rootPath=dict(kind='linear',velocity=3.5,direction=[0,1]))
        sampled,receipt=sample_pose(source,rig,params)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            save_motion(root/'source.npz',source,rig);save_motion(root/'pose.npz',sampled,rig)
            receipt['sourceMotion']=dict(path=str(root/'source.npz'),sha256=file_hash(root/'source.npz'))
            write_json(root/'sampling.json',receipt)
            delivered=freeze(sampled,0,25,rig);save_motion(root/'delivered.npz',delivered,rig)
            p=dict(file=str(root/'sampling.json'),sha256=file_hash(root/'sampling.json'),
                sourcePoseMotionFile=str(root/'pose.npz'),sourcePoseMotionSha256=file_hash(root/'pose.npz'),
                deliveredMotionSha256=file_hash(root/'delivered.npz'))
            job=dict(contract=dict(sampledPosePolicy=POSE_POLICY),poseSamplingReceipt=p,
                motionPath=str(root/'delivered.npz'),motionSha256=file_hash(root/'delivered.npz'))
            validate_pose_delivery(job)
            delivered['root_positions'][:,0]+=1
            save_motion(root/'delivered.npz',delivered,rig)
            job['motionSha256']=p['deliveredMotionSha256']=file_hash(root/'delivered.npz')
            with self.assertRaisesRegex(ValueError,'changed the sampled body'):validate_pose_delivery(job)

    def test_combined_pack_gets_separate_viewer_batch(self):
        from .full_pipeline import complete_run
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);pack=root/'pack';pack.mkdir()
            runner=SimpleNamespace(output=root,repo=root,python='python',graph=dict(pipeline='derived'),
                results={'asset':dict(op='deliver',identity='new',outputs={})})
            conversion=dict(viewer=dict(id='combined-target'),sourceClosureComplete=False,
                failedTargetGaits=[],failedTargetMantles=[],finalFbxReadback=dict(passed=True))
            with patch('tools.kimodo_pipeline.derived.retain_base_deliveries',side_effect=lambda p,a,r:(a,r)), \
                 patch('tools.kimodo_pipeline.offline_pipeline.convert_bundle',return_value=conversion) as convert, \
                 patch('tools.kimodo_pipeline.full_pipeline.verify_runtime',return_value=dict(status='runtime_passed')):
                result=complete_run(runner,dict(status='authoring_complete',viewerBatch=dict(id='new-source')),
                    base_pack=root/'base.json',pack=pack)
            self.assertIsNone(convert.call_args.kwargs['batch_id'])
            provenance=read_json(pack/'pipeline-provenance.json')
            self.assertIsNone(provenance['batchId'])
            self.assertEqual(provenance['targetViewerBatchId'],'combined-target')
            self.assertEqual(provenance['authoringBatchId'],'new-source')
            self.assertFalse(result['activated'])

    def test_base_pack_cannot_certify_zero_new_deliveries(self):
        from .full_pipeline import complete_run
        with tempfile.TemporaryDirectory() as directory:
            runner=SimpleNamespace(output=Path(directory),results={},graph=dict(pipeline='derived'))
            result=complete_run(runner,dict(status='authoring_rejected'),base_pack=Path(directory)/'missing-pack.json')
            self.assertEqual(result['failedPhase'],'source-quality')
            self.assertEqual(result['sourceDeliveries'],0)
            self.assertFalse(result['activated'])

    def test_retained_deliveries_merge_without_discarding_existing_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            base=dict(node='existing',identity='base',outputs={})
            old=dict(node='replace',identity='old',outputs={})
            new=dict(node='replace',identity='new',outputs={})
            rig=dict(outputs={'rig.json':dict(sha256='same-rig')})
            write_json(root/'authoring.json',dict(assets=[base,old]))
            write_json(root/'receipts.json',{'rig:soma':rig,'existing':dict(identity='base',outputs={}),
                'replace':dict(identity='old',outputs={})})
            write_json(root/'retarget.json',dict(authoringBundle=str(root/'authoring.json')))
            pack=root/'assets/generated/als_nonprop_test/resource-pack.json'
            pack.parent.mkdir(parents=True)
            write_json(pack,dict(files=[],nativeRetargetBundle=str(root/'retarget.json')))
            assets,receipts=retain_base_deliveries(pack,[new],{'rig:soma':rig,'replace':dict(identity='new')})
            self.assertEqual(assets,[dict(base,sourceAuthoringReceipts=dict(
                file=str(root/'receipts.json'),sha256=file_hash(root/'receipts.json'))),new])
            self.assertEqual(receipts['existing']['identity'],'base')
            self.assertEqual(receipts['replace']['identity'],'new')

    def test_sample_removes_path_preserves_height_and_exact_rotations(self):
        rig=rig_fixture();source=clip(rig,90)
        source['root_positions'][:,2]=3.5*np.arange(90)/30+.03
        source['root_positions'][:,0]=.02*np.sin(np.arange(90))
        source['root_positions'][:,1]=1+.01*np.cos(np.arange(90))
        before=copy.deepcopy(source)
        p=dict(policy=POSE_POLICY,semantic='Lean_Pose',parentSemantic='Run_Forward',phase=.25,
            periodSeconds=.8,samplingMode='conditioned-cycle',count=1,seconds=3,
            rootPath=dict(kind='linear',velocity=3.5,direction=[0,1]))
        pose,receipt=sample_pose(source,rig,p)
        self.assertEqual(receipt['sourceFrame'],30)
        np.testing.assert_array_equal(pose['local_rot_mats'][0],source['local_rot_mats'][30])
        self.assertEqual(pose['root_positions'][0,1],source['root_positions'][30,1])
        self.assertAlmostEqual(pose['root_positions'][0,2],0)
        for key in source:np.testing.assert_array_equal(source[key],before[key])
        p['periodSeconds']=10
        with self.assertRaisesRegex(ValueError,'out of range'):sample_pose(source,rig,p)

    def test_default_samples_do_not_depend_on_failed_loop_but_gaits_keep_gates(self):
        graph=build_graph(read_json(ROOT/'docs/architecture/kimodo-als-asset-contracts.json'),
            load_rules(PIPELINE_ROOT/'tools/kimodo_pipeline/directional_gait_rules.json'))
        order=topological_order(graph['nodes'],['asset:v4:ALS_N_Run_BasePose'])
        self.assertIn('candidate:Run_Forward',order)
        self.assertNotIn('motion:Run_Forward',order)
        self.assertNotIn('repair:Run_Forward:generated',order)
        self.assertEqual(graph['nodes']['motion:Run_Left_Forward']['op'],'gait_cycle')
        self.assertIsNone(graph['nodes']['asset:v4:ALS_N_Run_BasePose']['params']['targetPlanarOrigin'])
        contract=graph['nodes']['asset:v4:ALS_N_Run_BasePose']['params']['contract']
        self.assertEqual(contract['sampledPosePolicy'],POSE_POLICY)

    def test_reuse_changes_are_detected_and_unavailable_sources_never_generate(self):
        rig=rig_fixture()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);write_json(root/'rig.json',rig)
            graph=dict(nodes={'rig:soma':dict(id='rig:soma',op='rig',deps=[],params={}),
                'candidate':dict(id='candidate',op='generate',deps=['rig:soma'],params=dict(semantic='Run',rootPath=None))},
                rules=dict(fps=30),rulesDigest='test')
            runner=Runner(graph,root/'first',bindings={'rig:soma':dict(path=str(root/'rig.json'))})
            runner.output.mkdir()
            runner.execute_node('rig:soma')
            # Use the runner's real content-addressed receipt structure.
            from .motion import save_motion
            from .runner import file_hash
            save_motion(root/'source.npz',clip(rig),rig)
            receipt=dict(node='candidate',op='generate',identity='source',boundInput=False,
                inputs=dict(node=graph['nodes']['candidate'],dependencyOutputs={'rig:soma':{'rig.json':runner.results['rig:soma']['outputs']['rig.json']['sha256']}}),
                outputs={'motion.npz':dict(path=str(root/'source.npz'),sha256=file_hash(root/'source.npz'))})
            receipt['identity']=digest(receipt['inputs'])
            write_json(root/'receipts.json',{'candidate':receipt})
            imported=reuse_upstream(graph,[root/'receipts.json'],derived_only=True)
            next_runner=Runner(imported,root/'next',bindings={'rig:soma':dict(path=str(root/'rig.json'))})
            next_runner.output.mkdir()
            next_runner.execute_node('rig:soma');next_runner.execute_node('candidate')
            self.assertEqual(next_runner.results['candidate']['op'],'reuse_motion')
            (root/'source.npz').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'Changed upstream'):reuse_upstream(graph,[root/'receipts.json'])
            missing=reuse_upstream(graph,[],derived_only=True)
            self.assertEqual(missing['nodes']['candidate']['op'],'external')


if __name__=='__main__':unittest.main()

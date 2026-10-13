"""Resume and explicit feedback must fail closed before model loading or export."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from .graph import digest
from .runner import file_hash,write_json
from .mantle_refinement import PhaseProgress,author_failure_score,checked_feedback,feedback_options,refine,resume,verify_receipts


class MantleRefinementTests(unittest.TestCase):
    def test_quality_phase_changes_reset_stagnation_comparison(self):
        progress=PhaseProgress()
        progress.observe('source',1.1)
        self.assertEqual(progress.observe('target',6.)['consecutiveNonImprovingRounds'],0)
        self.assertEqual(progress.observe('target',5.)['consecutiveNonImprovingRounds'],0)
        self.assertEqual(progress.observe('source',2.)['consecutiveNonImprovingRounds'],0)
        self.assertEqual(progress.observe('source',1.9)['consecutiveNonImprovingRounds'],0)
        self.assertEqual(progress.observe('source',1.2)['consecutiveNonImprovingRounds'],0)
        self.assertEqual(progress.observe('source',1.2)['consecutiveNonImprovingRounds'],1)
        self.assertEqual(progress.observe('source',1.3)['consecutiveNonImprovingRounds'],2)

    def feedback(self,root,conditions=None):
        conditions=conditions or [dict(type='fullbody',frame_indices=[0,30],local_joints_rot=[],root_positions=[])]
        write_json(root/'constraints.json',conditions)
        write_json(root/'projection.json',dict(passed=True,deliverable=False,constraintsSha256=digest(conditions)))
        (root/'generated.npz').write_bytes(b'controlled-generated-payload')
        return {key:dict(file=str(root/name),sha256=file_hash(root/name)) for key,name in (
            ('constraints','constraints.json'),('projectionReceipt','projection.json'),('parentGeneratedMotion','generated.npz'))}

    def test_modified_parent_or_constraints_are_rejected_before_generation(self):
        for filename in ('generated.npz','constraints.json','projection.json'):
            with self.subTest(filename=filename),tempfile.TemporaryDirectory() as temporary:
                root=Path(temporary);feedback=self.feedback(root)
                checked_feedback(feedback)
                (root/filename).write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError,'input changed'):
                    checked_feedback(feedback)

    def test_resigning_a_condition_file_does_not_bypass_projection_binding(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);feedback=self.feedback(root)
            data=json.loads((root/'constraints.json').read_text(encoding='utf-8'))
            data[0]['frame_indices']=[1,30];write_json(root/'constraints.json',data)
            feedback['constraints']['sha256']=file_hash(root/'constraints.json')
            with self.assertRaisesRegex(ValueError,'does not bind'):
                checked_feedback(feedback)

    def test_union_budget_is_enforced_across_multiple_constraint_types(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            conditions=[dict(type='fullbody',frame_indices=list(range(11))),
                        dict(type='left-hand',frame_indices=list(range(10,21)))]
            with self.assertRaisesRegex(ValueError,'sparse'):
                checked_feedback(self.feedback(root,conditions))

    def test_full_temporal_budget_switches_to_existing_event_keys(self):
        conditions=[dict(type='fullbody',frame_indices=list(range(19)))]
        options=feedback_options(5,conditions)
        self.assertTrue(options['reuse_existing_time_keys'])
        self.assertEqual(options['maximum_added_frames'],1)

    def test_failure_score_uses_original_gate_ratios_without_replacing_the_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);report=dict(checks=[dict(passed=False,value=18.,limit=15.),dict(passed=False,value=3.,limit=2.),
                dict(passed=True,value=.5,limit=1.)])
            write_json(root/'mantle-validation.json',report)
            runner=SimpleNamespace(results={},events=[dict(node='motion:Mantle_High',status='failed',evidence=str(root))])
            self.assertEqual(author_failure_score(runner,'Mantle_High'),1.5)
            self.assertEqual(json.loads((root/'mantle-validation.json').read_text(encoding='utf-8')),report)
            runner.results['motion:Mantle_High']={}
            self.assertEqual(author_failure_score(runner,'Mantle_High'),0.)

    def test_resume_rejects_changed_graph_payload_and_input_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);path=root/'motion.npz';path.write_bytes(b'generated')
            node=dict(id='candidate:Mantle_High',op='generate',deps=[],params=dict(seed=42))
            graph=dict(nodes={node['id']:node});inputs=dict(node=node,runtime='controlled')
            receipt=dict(node=node['id'],op='generate',inputs=inputs,identity=digest(inputs),
                outputs=dict(motion=dict(path=str(path),sha256=file_hash(path))))
            verify_receipts(graph,{node['id']:receipt})
            receipt['identity']='changed'
            with self.assertRaisesRegex(ValueError,'graph'):
                verify_receipts(graph,{node['id']:receipt})
            receipt['identity']=digest(inputs);path.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'payload'):
                verify_receipts(graph,{node['id']:receipt})

    def test_resume_requires_rule_identity_batch_identity_and_dependency_closure(self):
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'payload';path.write_bytes(b'controlled')
            source=dict(id='source',op='generate',deps=[],params={})
            delivery=dict(id='delivery',op='deliver',deps=['source'],params={})
            graph=dict(nodes=dict(source=source,delivery=delivery),rules=dict(fps=30),rulesDigest=digest(dict(fps=30)))
            outputs=dict(motion=dict(path=str(path),sha256=file_hash(path)))
            source_inputs=dict(node=source,rules=graph['rulesDigest'],dependencies={},dependencyOutputs={})
            source_receipt=dict(node='source',op='generate',inputs=source_inputs,identity=digest(source_inputs),outputs=outputs)
            delivery_inputs=dict(node=delivery,rules=graph['rulesDigest'],dependencies=dict(source=source_receipt['identity']),
                dependencyOutputs=dict(source={key:value['sha256'] for key,value in outputs.items()}))
            delivery_receipt=dict(node='delivery',op='deliver',inputs=delivery_inputs,identity=digest(delivery_inputs),outputs=outputs)
            receipts=dict(source=source_receipt,delivery=delivery_receipt)
            batch=dict(graphDigest=digest(graph));verify_receipts(graph,receipts,batch)
            with self.assertRaisesRegex(ValueError,'dependency receipt missing'):
                verify_receipts(graph,dict(delivery=delivery_receipt),batch)
            with self.assertRaisesRegex(ValueError,'batch'):
                verify_receipts(graph,receipts,dict(graphDigest='changed'))
            graph['rules']['fps']=99
            with self.assertRaisesRegex(ValueError,'rules digest'):
                verify_receipts(graph,receipts,batch)

    def test_resume_preserves_partial_output_and_forwards_new_options_to_a_fresh_continuation(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary);author_dir=folder/'authoring';author_dir.mkdir()
            prior=author_dir/'full-pipeline';prior.mkdir();(prior/'keep.log').write_text('interrupted evidence')
            graph=dict(nodes={});batch=dict(status='authoring_complete',graphDigest=digest(graph))
            write_json(folder/'graph.json',graph);write_json(author_dir/'receipts.json',{});write_json(author_dir/'batch-result.json',batch)
            class FakeRunner:
                def __init__(self,graph,output,*unused):
                    self.graph=graph;self.output=Path(output);self.results={}
            calls=[]
            def complete(runner,authoring,**options):
                self.assertFalse((runner.output/'full-pipeline').exists())
                calls.append((runner.output,options))
                return dict(status='pipeline_complete_rejected',productionReady=False)
            with patch('tools.kimodo_pipeline.mantle_refinement.MantleRefinementRunner',FakeRunner),patch(
                    'tools.kimodo_pipeline.full_pipeline.complete_run',complete):
                result=resume(folder,base_pack='new-base.json',runtime_cases=['new-case'])
            self.assertEqual((prior/'keep.log').read_text(),'interrupted evidence')
            self.assertEqual(calls[0][1]['base_pack'],'new-base.json')
            self.assertEqual(calls[0][1]['runtime_cases'],['new-case'])
            self.assertEqual(Path(result['resumeContinuation']),(folder/'continuations/run-001').resolve())

    def test_existing_output_is_preserved_and_resume_requires_real_receipts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);sentinel=root/'keep.txt';sentinel.write_text('keep')
            with self.assertRaisesRegex(ValueError,'already exists'):
                refine(rules_path='unused',semantic='Mantle_High',output=root)
            with self.assertRaisesRegex(ValueError,'receipts'):
                resume(root)
            self.assertEqual(sentinel.read_text(),'keep')


if __name__=='__main__':
    unittest.main()

from .paths import project_root, asset_root, PIPELINE_ROOT
"""Crouch closure uses its authored directions, time domains and native geometry."""
import copy
import unittest
from pathlib import Path

from .graph import build_graph, load_rules, read_json, topological_order
from .scope import crouch_targets
from .target_gait import analyze_target_gait, original_protocol

ROOT=asset_root()


class CrouchPipelineTests(unittest.TestCase):
    def test_production_front_back_crouch_preserves_reference_heading_only_in_requested_lanes(self):
        graph = build_graph(read_json(ROOT/'docs/architecture/kimodo-als-asset-contracts.json'),
                            load_rules(PIPELINE_ROOT/'tools/kimodo_pipeline/crouch_rules.json'))
        for name in ('Crouch_Walk_Forward', 'Crouch_Walk_Backward'):
            params = graph['nodes']['candidate:'+name]['params']
            self.assertTrue(params['preserveReferenceHeading'])
            self.assertEqual(20261011, params['seed'])
            self.assertEqual('reference:'+name+':native-als', params['gaitReference'])
            self.assertEqual(1.5, params['rootPath']['velocity'])
        self.assertNotIn('preserveReferenceHeading', graph['nodes']['candidate:Walk_Forward']['params'])
        for edition, asset in (('v4','ALS_CLF_Walk_B'), ('refactored','A_Als_Crouch_Walk_Backward')):
            self.assertEqual(2, graph['nodes']['asset:'+edition+':'+asset]['params']['contract']['targetPhaseHarmonics'])
        rules = load_rules(PIPELINE_ROOT/'tools/kimodo_pipeline/crouch_rules.json')
        rules['gaitClosure']['Crouch_Walk_Backward']['targetPhaseHarmonics'] = True
        with self.assertRaisesRegex(ValueError, 'Target phase harmonics'):
            build_graph(read_json(ROOT/'docs/architecture/kimodo-als-asset-contracts.json'), rules)

    def test_scope_closes_missing_cycles_without_pose_air_or_prop_generation(self):
        contracts=read_json(ROOT/'docs/architecture/kimodo-als-asset-contracts.json')
        graph=build_graph(contracts,load_rules(PIPELINE_ROOT/'tools/kimodo_pipeline/crouch_rules.json'))
        targets=crouch_targets(contracts)
        self.assertEqual(6,len(targets))
        order=topological_order(graph['nodes'],targets)
        self.assertFalse(any(graph['nodes'][key]['op']=='external' for key in order))
        self.assertFalse(any(key.startswith('candidate:Run') or key.startswith('candidate:Fall') for key in order))
        self.assertNotIn('candidate:Crouch_Walk_Right',order)
        for key in targets:
            self.assertEqual('als-forward-v1',graph['nodes'][key]['params']['contract']['targetGaitPolicy'])

    def test_native_crouch_geometry_and_direction_are_measured(self):
        protocol=original_protocol()
        for kind,direction in (('crouch',(0.,1.)),('crouch_backward',(0.,-1.)),
                               ('crouch_left',(1.,0.)),('crouch_right',(-1.,0.))):
            with self.subTest(kind=kind):
                row=protocol['assets'][kind]
                report=analyze_target_gait(row['raw'],row['skeleton'],row['source'],protocol)
                self.assertTrue(report['passed'],[c for c in report['checks'] if not c['passed']])
                self.assertEqual(direction,report['travelDirectionXY'])
                frozen=copy.deepcopy(row['raw'])
                for track in frozen['tracks']:
                    for field in ('positions','rotations','scales'):
                        if track[field]:track[field]=[track[field][0]]
                damaged=analyze_target_gait(frozen,row['skeleton'],row['source'],protocol)
                self.assertFalse(damaged['passed'])

    def test_right_master_can_produce_left_without_a_dependency_cycle(self):
        rules=load_rules(PIPELINE_ROOT/'tools/kimodo_pipeline/crouch_rules.json')
        rules['gaitMirrorSources']={'Crouch_Walk_Left':'Crouch_Walk_Right'}
        contracts=read_json(ROOT/'docs/architecture/kimodo-als-asset-contracts.json')
        graph=build_graph(contracts,rules)
        order=topological_order(graph['nodes'],crouch_targets(contracts))
        self.assertIn('candidate:Crouch_Walk_Right',order)
        self.assertNotIn('candidate:Crouch_Walk_Left',order)
        self.assertEqual('motion:Crouch_Walk_Right',graph['nodes']['motion:Crouch_Walk_Left']['params']['source'])
        rules['gaitMirrorSources']['Crouch_Walk_Right']='Crouch_Walk_Left'
        with self.assertRaisesRegex(ValueError,'independent master'):build_graph(contracts,rules)

    def test_reused_mirror_master_still_enforces_the_requested_time_domain(self):
        import tempfile
        from .derived import reuse_upstream
        from .gait import author_cycle
        from .graph import digest
        from .motion import save_motion
        from .runner import Runner,write_json,file_hash
        from .test_gait import walking_fixture
        source,rig=walking_fixture(seconds=5)
        accepted,gate=author_cycle(source,rig,61,.6)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            write_json(root/'rig.json',rig)
            key='motion:Crouch_Walk_Right'
            graph=dict(nodes={'rig:soma':dict(id='rig:soma',op='rig',deps=[],params={}),
                key:dict(id=key,op='gait_cycle',deps=['rig:soma'],params=dict(targetCount=61))},
                rules=dict(fps=60),rulesDigest='test')
            binding={'rig:soma':dict(path=str(root/'rig.json'))}
            first=Runner(graph,root/'first',bindings=binding)
            first.output.mkdir();first.execute_node('rig:soma')
            save_motion(root/'motion.npz',accepted,rig)
            write_json(root/'gate.json',gate)
            write_json(root/'trajectory.json',gate['removedPlanarTrajectory'])
            inputs=dict(node=dict(id=key,op='mirror',deps=['rig:soma'],params={}),
                dependencyOutputs={'rig:soma':{'rig.json':first.results['rig:soma']['outputs']['rig.json']['sha256']}})
            receipt=dict(node=key,op='mirror',identity=digest(inputs),inputs=inputs,boundInput=False,
                outputs={name:dict(path=str(root/file),sha256=file_hash(root/file)) for name,file in
                         (('motion.npz','motion.npz'),('gait-validation.json','gate.json'),
                          ('authoring-trajectory.json','trajectory.json'))})
            write_json(root/'receipts.json',{key:receipt})
            imported=reuse_upstream(graph,[root/'receipts.json'])
            next_runner=Runner(imported,root/'next',bindings=binding)
            next_runner.output.mkdir();next_runner.execute_node('rig:soma');next_runner.execute_node(key)
            self.assertEqual(file_hash(root/'motion.npz'),next_runner.results[key]['outputs']['motion.npz']['sha256'])
            graph['nodes'][key]['params']['targetCount']=62
            wrong=Runner(reuse_upstream(graph,[root/'receipts.json']),root/'wrong',bindings=binding)
            wrong.output.mkdir();wrong.execute_node('rig:soma')
            with self.assertRaisesRegex(ValueError,'time domain differs'):wrong.execute_node(key)

    def test_denser_pose_conditions_preserve_the_common_constraint_budget(self):
        from .gait_constraints import materialize_gait_constraints,GaitConstraintError
        from .test_gait_constraints import parameters,decode
        from .test_gait import walking_fixture
        import numpy as np
        rules=load_rules(PIPELINE_ROOT/'tools/kimodo_pipeline/crouch_rules.json')
        phases=[.125,.25,.375,.5,.625,.75,.875]
        rules['gaitClosure']['Crouch_Walk_Forward'].update(fullBodyPhaseFractions=phases,fullBodyPhaseCycles=2,
                                                        boundaryContextFrames=0)
        graph=build_graph(read_json(ROOT/'docs/architecture/kimodo-als-asset-contracts.json'),rules)
        configured=graph['nodes']['candidate:Crouch_Walk_Forward']['params']
        source,rig=walking_fixture(seconds=5)
        params=parameters()
        params.update(seconds=3.02,fullBodyPhaseFractions=configured['fullBodyPhaseFractions'],
                      fullBodyPhaseCycles=configured['fullBodyPhaseCycles'],maximumConstraintCycles=3,
                      boundaryContextFrames=configured['boundaryContextFrames'],supportAnchorCount=4)
        constraints,receipt=materialize_gait_constraints(params,source,rig)
        self.assertTrue(receipt['pass'])
        self.assertLess(max(receipt['constraintCounts'].values()),20)
        poses=decode(next(c for c in constraints if c['type']=='fullbody'),rig)
        for frame in (8,22,38,52):
            np.testing.assert_allclose(poses[frame][0],poses[frame+60][0],atol=1e-12)
        params['boundaryContextFrames']=1
        with self.assertRaisesRegex(GaitConstraintError,'budget exceeded'):
            materialize_gait_constraints(params,source,rig)


if __name__=='__main__':
    unittest.main()

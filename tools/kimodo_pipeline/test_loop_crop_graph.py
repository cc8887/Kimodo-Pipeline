import copy
import unittest

from .loop_crop_graph import configure
from .graph import topological_order


class LoopCropGraphTests(unittest.TestCase):
    def nodes(self):
        return {'rig:soma': dict(id='rig:soma', op='rig', deps=[], params={}),
            'source': dict(id='source', op='generate', deps=['rig:soma'], params={}),
            'idle': dict(id='idle', op='copy', deps=['rig:soma'], params={}),
            'reference': dict(id='reference', op='native_gait_reference', deps=['rig:soma'], params=dict(nativePhaseEvidence=True)),
            'motion:Sprint': dict(id='motion:Sprint', op='gait_cycle', deps=['rig:soma', 'source', 'idle'],
                params=dict(source='source', targetCount=19, expectedSpeedMps=6.,
                            postureValidationIdle='idle', postureProfile={'policy': 'example'}))}

    def spec(self):
        return dict(loopCrops={'Sprint': dict(startFrame=18, endFrameInclusive=36,
                   contactReference='reference', jointContactSeam=True)})

    def test_single_cycle_enters_normal_delivery_order_after_final_style(self):
        nodes = self.nodes(); before = copy.deepcopy(nodes['source']); configure(nodes, self.spec())
        crop = nodes['loop-crop:Sprint']; master = nodes['motion:Sprint']
        self.assertEqual(before, nodes['source'])
        self.assertEqual('loop-crop:Sprint', master['params']['source'])
        self.assertTrue(master['params']['postureAlreadyFinal'])
        self.assertEqual(6., crop['params']['expectedSpeedMps'])
        self.assertEqual({'policy': 'example'}, crop['params']['postureProfile'])
        order = topological_order(nodes, ['motion:Sprint'])
        self.assertLess(order.index('reference'), order.index('loop-crop:Sprint'))
        self.assertLess(order.index('loop-crop:Sprint'), order.index('motion:Sprint'))

    def test_time_speed_reference_and_cyclic_source_are_not_overrideable(self):
        for patch in ({'endFrameInclusive': 35}, {'expectedSpeedMps': 5.}, {'source': 'motion:Sprint'},
                      {'contactReference': 'source'}, {'source': 'missing'}):
            rules = self.spec(); rules['loopCrops']['Sprint'].update(patch)
            with self.subTest(patch=patch), self.assertRaises(ValueError):
                configure(self.nodes(), rules)

    def test_joint_repair_cannot_omit_independent_phase_evidence(self):
        nodes = self.nodes(); nodes['reference']['params']['nativePhaseEvidence'] = False
        with self.assertRaises(ValueError):
            configure(nodes, self.spec())
        rules = self.spec(); rules['loopCrops']['Sprint'].pop('contactReference')
        with self.assertRaises(ValueError):
            configure(self.nodes(), rules)

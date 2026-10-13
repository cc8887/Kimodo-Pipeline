import copy
import unittest

from .mantle import PROFILES, materialize_constraints, native_reference
from .mantle_constraints import contact_plan
from .test_mantle import soma_rig


class SparseMantleConstraintTests(unittest.TestCase):
    def test_sparse_conditions_preserve_contact_tasks_without_pinning_intermediate_body_pose(self):
        rig = soma_rig()
        for name in PROFILES:
            guide, _ = native_reference(name, rig)
            params = dict(semantic=name, seconds=3, fps=30)
            old, _ = materialize_constraints(params, guide, rig)
            sparse, receipt = materialize_constraints(dict(params, mantleConstraintPolicy='sparse-contact-v1'), guide, rig)
            kinds = {c['type']: c for c in sparse}
            self.assertEqual(set(kinds), {c['type'] for c in old})
            self.assertEqual(kinds['fullbody']['frame_indices'], [0,89])
            self.assertGreater(len(old[0]['frame_indices']), 2)
            self.assertLess(len(set(f for c in sparse for f in c['frame_indices'])), 20)
            for side in PROFILES[name]['hands']:
                self.assertEqual(len(kinds[side+'-hand']['frame_indices']), 5)
            self.assertEqual(receipt['contactPlan']['policy'], 'sparse-contact-v1')
            self.assertFalse(receipt['productionReady'])
            self.assertEqual(sparse, materialize_constraints(dict(params, mantleConstraintPolicy='sparse-contact-v1'),guide,rig)[0])

    def test_invalid_events_and_unknown_policy_fail_instead_of_falling_back(self):
        invalid = copy.deepcopy(PROFILES['Mantle_High'])
        invalid['handContactSeconds']['left'] = (2., 1.)
        with self.assertRaisesRegex(ValueError, 'outside'):
            contact_plan(invalid, 3., 90)
        invalid = copy.deepcopy(PROFILES['Mantle_High'])
        invalid['hands'] = ('left', 'left')
        with self.assertRaisesRegex(ValueError, 'repeated'):
            contact_plan(invalid, 3., 90)
        with self.assertRaisesRegex(ValueError, 'Unknown Mantle'):
            materialize_constraints(dict(mantleConstraintPolicy='silent-fallback'), None, None)

    def test_plan_is_time_scaled_and_does_not_depend_on_style_text(self):
        short = contact_plan(PROFILES['Mantle_High'], 2.6666667, 81)
        long = contact_plan(PROFILES['Mantle_High'], 2.6666667, 161)
        self.assertEqual(short['contacts']['left']['beginSeconds'], long['contacts']['left']['beginSeconds'])
        self.assertEqual(short['contacts']['left']['keyFrames'][0]*2, long['contacts']['left']['keyFrames'][0])
        self.assertEqual(long['fullBodyFrames'], [0,160])


if __name__ == '__main__':
    unittest.main()

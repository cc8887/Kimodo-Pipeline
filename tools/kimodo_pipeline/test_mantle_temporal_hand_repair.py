"""Interpolation and nonconvex obstacle geometry contracts used by the repair."""
import unittest
import numpy as np
from scipy.spatial.transform import Rotation,Slerp
from .mantle_temporal_hand_repair import nlerp,pareto_escape


class MantleTemporalGeometryTests(unittest.TestCase):
    def test_runtime_nlerp_is_not_constant_angular_speed_slerp(self):
        endpoints=Rotation.from_euler('z',[[0.],[150.]],degrees=True)
        actual=nlerp(endpoints.as_matrix()[:1],endpoints.as_matrix()[1:],np.array([.25]))
        q=endpoints.as_quat();expected=Rotation.from_quat(q[0]*.75+q[1]*.25).as_matrix()
        np.testing.assert_allclose(actual[0],expected,atol=1e-12)
        self.assertGreater(np.linalg.norm(actual[0]-Slerp([0,1],endpoints)([.25]).as_matrix()[0]),.05)

    def test_shortest_arc_crosses_angle_wrap(self):
        endpoints=Rotation.from_euler('z',[[170.],[-170.]],degrees=True).as_matrix()
        actual=nlerp(endpoints[:1],endpoints[1:],np.array([.5]))
        np.testing.assert_allclose(actual[0],Rotation.from_euler('z',180,degrees=True).as_matrix(),atol=1e-12)

    def test_pareto_corner_escape_uses_top_or_front_union(self):
        points=np.array([[0,54,191],[0,59,196]])
        displacement=pareto_escape(points,200,clearance=0.)
        np.testing.assert_allclose(displacement,[0,-4,4])
        moved=points+displacement
        self.assertTrue(np.all(np.maximum(moved[:,2]-200,50-moved[:,1])>=0))
        self.assertLess(np.linalg.norm(displacement),9.)


if __name__=='__main__':unittest.main()

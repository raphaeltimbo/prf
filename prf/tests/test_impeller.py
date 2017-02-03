import pytest
import numpy as np
from prf.state import *
from prf.curve import *
from prf.impeller import *
from numpy.testing import assert_allclose


@pytest.fixture
def impeller1():
    fluid = {'CarbonDioxide': 0.79585,
             'R134a': 0.16751,
             'Nitrogen': 0.02903,
             'Oxygen': 0.00761}
    units = {'p_units': 'bar', 'T_units': 'degK', 'speed_units': 'RPM',
             'flow_m_units': 'kg/h'}
    suc = State.define(fluid=fluid, p=2.238, T=298.3, **units)
    disch = State.define(fluid=fluid, p=7.255, T=391.1, **units)

    point = Point(suc=suc, disch=disch, speed=7941, flow_m=34203.6)
    b = 0.0285
    D = 0.365
    return Impeller([point], b, D)


def test_impeller1(impeller1):
    non_dim_point_0 =  np.array([[  3.26149996e-04,   3.01619832e-04],
                                 [  2.45924494e+02,   2.90347051e+02],
                                 [  7.58596759e-01,   8.16683236e-01]])
    flow_coeff0 = 3.26149996e-04
    head_coeff0 = 2.45924494e+02
    eff0 = 7.58596759e-01
    #assert_allclose(impeller1.non_dim_points[0].flow_coeff, flow_coeff0)
    assert_allclose(impeller1.non_dim_points[0].head_coeff, head_coeff0)
    assert_allclose(impeller1.non_dim_points[0].eff, eff0)


@pytest.mark.skip
def test_mach(impeller1):
    mac = 0.06700529708077983
    assert_allclose(impeller1.mach(impeller1.curves[0].suc[0],
                                   impeller1.curves[0].speed[0]),
                    mac)

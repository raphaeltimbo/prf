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

    point = Point(suc=suc, disch=disch, speed=7941, flow_m=34203.6, **units)
    b = 0.0285
    D = 0.365
    return Impeller([point], b, D)


def test_impeller1(impeller1):
    flow_coeff0 = 3.26149996e-04
    head_coeff0 = 2.45924494e+02
    eff0 = 7.58596759e-01
    assert_allclose(impeller1.non_dim_points[0].flow_coeff, flow_coeff0)
    assert_allclose(impeller1.non_dim_points[0].head_coeff, head_coeff0)
    assert_allclose(impeller1.non_dim_points[0].eff, eff0)


def test_mach(impeller1):
    mach = 0.6398534546248233
    assert_allclose(impeller1.mach(impeller1.points[0].suc,
                                   impeller1.points[0].speed),
                    mach)

import pytest
import numpy as np
from prf.state import *
from prf.curve import *
from prf.impeller import *
from numpy.testing import assert_allclose


@pytest.fixture
def impeller():
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


def test_impeller0(impeller):
    flow_coeff = 0.12295565184151291
    head_coeff = 5.393727708502117
    assert_allclose(impeller.flow_coeff(point=0), flow_coeff)
    assert_allclose(impeller.head_coeff(point=0), head_coeff)


def test_impeller1(impeller):
    flow_coeff0 = 0.12295565184151291
    head_coeff0 = 5.393727708502117
    eff0 = 7.58596759e-01
    assert_allclose(impeller.non_dim_points[0].flow_coeff, flow_coeff0)
    assert_allclose(impeller.non_dim_points[0].head_coeff, head_coeff0)
    assert_allclose(impeller.non_dim_points[0].eff, eff0)


def test_mach(impeller):
    mach = 0.6398534546248233
    assert_allclose(impeller.mach(impeller.points[0].suc,
                                  impeller.points[0].speed),
                    mach)

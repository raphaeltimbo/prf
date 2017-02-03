import pytest
from prf.state import *
from prf.curve import *
from numpy.testing import assert_allclose


@pytest.fixture
def suc_1():
    fluid = {'CarbonDioxide': 0.76064,
             'R134a': 0.23581,
             'Nitrogen': 0.00284,
             'Oxygen': 0.00071}
    units = {'p_units': 'bar', 'T_units': 'degK', }

    return State.define(fluid=fluid, p=1.839, T=291.5, **units)


@pytest.fixture
def disch_1():
    fluid = {'CarbonDioxide': 0.76064,
             'R134a': 0.23581,
             'Nitrogen': 0.00284,
             'Oxygen': 0.00071}
    units = {'p_units': 'bar', 'T_units': 'degK'}

    return State.define(fluid=fluid, p=5.902, T=380.7, **units)


def test_n_exp(suc_1, disch_1):
    assert_allclose(n_exp(suc_1, disch_1), 1.2911165566270133)


def test_head_pol(suc_1, disch_1):
    assert_allclose(head_pol(suc_1, disch_1), 55280.81425974899)


def test_ef_pol(suc_1, disch_1):
    assert_allclose(eff_pol(suc_1, disch_1), 0.7112136965155706)


def test_head_isen(suc_1, disch_1):
    assert_allclose(head_isen(suc_1, disch_1), 53166.29655014263)


def test_eff_isen(suc_1, disch_1):
    assert_allclose(eff_isen(suc_1, disch_1), 0.68400943086328725)


def test_schultz_f(suc_1, disch_1):
    assert_allclose(schultz_f(suc_1, disch_1), 1.0017478040647996)


def test_head_pol_schultz(suc_1, disch_1):
    assert_allclose(head_pol_schultz(suc_1, disch_1), 55377.434270913633)


def test_point(suc_1, disch_1):
    point = Point(suc=suc_1, disch=disch_1, speed=7666, flow_m=29833.2)
    assert point.suc == suc_1
    assert point.disch == disch_1
    assert_allclose(point.flow_m, 29833.2)
    assert_allclose(point.speed, 7666)


def test_point_calc_1(suc_1, disch_1):
    units = {'p_units': 'bar',
             'T_units': 'degK',
             'speed_units': 'RPM',
             'flow_m_units': 'kg/h'}
    point = Point(suc=suc_1, disch=disch_1, speed=7666, flow_m=29833.2, **units)
    assert_allclose(point.head, 55377.434270913633)
    assert_allclose(point.eff, 0.7112136965155706)
    assert_allclose(point.flow_m, 8.287)
    assert_allclose(point.speed, 802.7816427473118)



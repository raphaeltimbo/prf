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


def test_ef_pol_schultz(suc_1, disch_1):
    assert_allclose(eff_pol_schultz(suc_1, disch_1), 0.712456758705285)


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
    assert_allclose(point.eff, 0.712456758705285)
    assert_allclose(point.flow_m, 8.287)
    assert_allclose(point.speed, 802.7816427473118)


@pytest.fixture
def state_si_main_op():
    fluid = {'Methane': 0.69945,
             'Ethane': 0.09729,
             'Propane': 0.05570,
             'n-Butane': 0.01780,
             'Isobutane': 0.01020,
             'n-Pentane': 0.00390,
             'Isopentane': 0.00360,
             'n-Hexane': 0.00180,
             'Nitrogen': 0.01490,
             'HydrogenSulfide': 0.00017,
             'CarbonDioxide': 0.09259,
             'Water': 0.00200}
    units = {'p_units': 'bar', 'T_units': 'degC'}

    return (State.define(p=16.99, T=38.4, fluid=fluid, **units),  # suc
            140349.53763396584,                                   # head
            0.71121369651557265)                                  # eff


def test_point_calc_from_suc_head_eff(state_si_main_op):
    suc, head, eff = (state_si_main_op)
    units = {'flow_m_units': 'kg/h', 'speed_units': 'RPM'}
    point = Point(suc=suc, head=head, eff=eff, flow_m=175171, speed=12204, **units)
    assert_allclose(point.disch.p(), 5344345.616396286)
    assert_allclose(point.disch.T(), 417.73930487362134)
    assert_allclose(point.speed, 1277.9998914803277)
    assert_allclose(point.flow_m, 48.658611111111114)
    assert_allclose(point.head, head)

@pytest.fixture
def point_si_main_op():
    fluid = {'Methane': 0.69945,
             'Ethane': 0.09729,
             'Propane': 0.05570,
             'n-Butane': 0.01780,
             'Isobutane': 0.01020,
             'n-Pentane': 0.00390,
             'Isopentane': 0.00360,
             'n-Hexane': 0.00180,
             'Nitrogen': 0.01490,
             'HydrogenSulfide': 0.00017,
             'CarbonDioxide': 0.09259,
             'Water': 0.00200}
    units = {'p_units': 'bar', 'T_units': 'degC'}

    suc = State.define(p=16.99, T=38.4, fluid=fluid, **units)

    point = Point(suc=suc, head=140349.53763396584, eff=0.71121369651557265,
                  flow_m=175171, speed=12204, **units)

    return point


def test_point_calc_from_suc_eff_vol_ratio(point_si_main_op):
    p0 = point_si_main_op
    p1 = Point(suc=p0.suc, eff=p0.eff, volume_ratio=p0.volume_ratio,
               speed=p0.speed, flow_m=p0.flow_m)

    assert p0 != p1
    assert_allclose(p0.suc.T(), p1.suc.T())
    assert_allclose(p0.suc.p(), p1.suc.p())
    assert_allclose(p0.disch.T(), p1.disch.T())
    assert_allclose(p0.disch.p(), p1.disch.p())
    assert_allclose(p0.speed, p1.speed)
    assert_allclose(p0.flow_m, p1.flow_m)
    assert_allclose(p0.flow_v, p1.flow_v)
    assert_allclose(p0.eff, p1.eff)
    assert_allclose(p0.head, p1.head)
    assert_allclose(p0.power, p1.power)
    assert_allclose(p0.volume_ratio, p1.volume_ratio)

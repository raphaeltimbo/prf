import pytest
from prf.state import *
from prf.point import *
from numpy.testing import assert_allclose

skip = True  # skip slow tests


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


@pytest.fixture
def point_1():
    fluid = {'CarbonDioxide': 0.76064,
             'R134a': 0.23581,
             'Nitrogen': 0.00284,
             'Oxygen': 0.00071}
    units = {'p_units': 'bar', 'T_units': 'degK'}

    suc_ = State.define(fluid=fluid, p=1.839, T=291.5, **units)
    disch_ = State.define(fluid=fluid, p=5.902, T=380.7, **units)

    return Point(suc=suc_, disch=disch_, speed=7666, flow_m=29833.2)


def test_n_exp(point_1, suc_1, disch_1):
    assert_allclose(point_1.n_exp(suc=suc_1, disch=disch_1), 1.2910634059847939, rtol=1e-4)
    assert_allclose(point_1.n_exp(), 1.2910634059847939, rtol=1e-4)


def test_head_pol(point_1, suc_1, disch_1):
    assert_allclose(point_1.head_pol(suc=suc_1, disch=disch_1), 55280.82459048466, rtol=1e-4)
    assert_allclose(point_1.head_pol(), 55280.82459048466, rtol=1e-4)


def test_ef_pol(point_1, suc_1, disch_1):
    assert_allclose(point_1.eff_pol(suc=suc_1, disch=disch_1), 0.7111862811638862, rtol=1e-4)
    assert_allclose(point_1.eff_pol(), 0.7111862811638862, rtol=1e-4)


def test_ef_pol_schultz(point_1, suc_1, disch_1):
    assert_allclose(point_1.eff_pol_schultz(suc=suc_1, disch=disch_1), 0.7124304497904342, rtol=1e-4)
    assert_allclose(point_1.eff_pol_schultz(), 0.7124304497904342, rtol=1e-4)


def test_head_isen(point_1, suc_1, disch_1):
    assert_allclose(point_1.head_isen(suc=suc_1, disch=disch_1), 53166.12359933178, rtol=1e-4)
    assert_allclose(point_1.head_isen(), 53166.12359933178, rtol=1e-4)


def test_eff_isen(point_1, suc_1, disch_1):
    assert_allclose(point_1.eff_isen(suc=suc_1, disch=disch_1), 0.6839807113336114, rtol=1e-4)
    assert_allclose(point_1.eff_isen(), 0.6839807113336114, rtol=1e-4)


def test_schultz_f(point_1, suc_1, disch_1):
    assert_allclose(point_1.schultz_f(suc=suc_1, disch=disch_1), 1.0017494272028307, rtol=1e-4)
    assert_allclose(point_1.schultz_f(), 1.0017494272028307, rtol=1e-4)


def test_head_pol_schultz(point_1, suc_1, disch_1):
    assert_allclose(point_1.head_pol_schultz(suc=suc_1, disch=disch_1), 55377.53436881817, rtol=1e-4)
    assert_allclose(point_1.head_pol_schultz(), 55377.53436881817, rtol=1e-4)


def test_point(suc_1, disch_1):
    point = Point(suc=suc_1, disch=disch_1, speed=7666, flow_m=29833.2)
    assert point.suc == suc_1
    assert point.disch == disch_1
    assert_allclose(point.flow_m, 29833.2)
    assert_allclose(point.speed, 7666)
    assert_allclose(point.flow_v, 6724., rtol=1e-4)
    # create point with flow_v
    point = Point(suc=suc_1, disch=disch_1, speed=7666, flow_v=6724.)
    assert_allclose(point.flow_m, 29833.2, rtol=1e-4)

    with pytest.raises(Exception):
        Point(suc=suc_1, disch=disch_1, speed=7666.)


def test_point_calc_1(suc_1, disch_1):
    units = {'p_units': 'bar',
             'T_units': 'degK',
             'speed_units': 'RPM',
             'flow_m_units': 'kg/h'}
    point = Point(suc=suc_1, disch=disch_1, speed=7666, flow_m=29833.2, **units)
    assert_allclose(point.head, 55377.53436881817, rtol=1e-4)
    assert_allclose(point.eff, 0.7124304394025333, rtol=1e-4)
    assert_allclose(point.flow_m, 8.287, rtol=1e-4)
    assert_allclose(point.speed, 802.7816427473118, rtol=1e-4)


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
            140349.53763396584,  # head
            0.71121369651557265)                                  # eff


def test_point_calc_from_suc_head_eff(state_si_main_op):
    suc, head, eff = state_si_main_op
    units = {'flow_m_units': 'kg/h', 'speed_units': 'RPM'}
    point = Point(suc=suc, head=head, eff=eff, flow_m=175171, speed=12204, **units)
    assert_allclose(point.disch.p(), 5343655.164571259, rtol=1e-3)
    assert_allclose(point.disch.T(), 417.73930487362134, rtol=1e-3)
    assert_allclose(point.speed, 1277.9998914803277, rtol=1e-4)
    assert_allclose(point.flow_m, 48.658611111111114, rtol=1e-4)
    assert_allclose(point.head, head, rtol=1e-4)


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


@pytest.mark.skipif(skip is True, reason='Slow test')
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


def test_point_calc_from_suc_eff_vol_ratio(point_1):
    p0 = point_1
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


def test_point_calc_from_suc_head_power(point_1):
    p0 = point_1
    p1 = Point(suc=p0.suc, head=p0.head, power=p0.power,
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


def test_curve(point_1):
    p0 = point_1
    p1 = Point(suc=p0.suc, eff=p0.eff, volume_ratio=p0.volume_ratio,
               speed=p0.speed, flow_m=p0.flow_m+1)
    flow_v = p0.flow_v
    curve = Curve([p0, p1])
    assert_allclose(p0.suc.p(), curve.suc_p_curve(flow_v))
    assert_allclose(p0.suc.T(), curve.suc_T_curve(flow_v))
    assert_allclose(p0.disch.p(), curve.disch_p_curve(flow_v))
    assert_allclose(p0.disch.T(), curve.disch_T_curve(flow_v))
    assert_allclose(p0.head, curve.head_curve(flow_v))
    assert_allclose(p0.eff, curve.eff_curve(flow_v))
    assert_allclose(p0.power, curve.power_curve(flow_v))
# TODO add tests for load curves

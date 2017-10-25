import pytest
import CoolProp.CoolProp as CP
from prf.state import *
from copy import copy
from numpy.testing import assert_allclose

skip = False  # skip slow tests


def test_convert_units():
    units = {
        'p_units': 'bar',
        'T_units': 'degC',
        'speed_units': 'RPM',
        'flow_m_units': 'kg/h',
        'flow_v_units': 'm**3/h',
        'power_units': 'kW'
        }

    @convert_to_base_units
    def func(**kwargs):
        return kwargs

    converted = func(p=16.99, T=30, speed=1000, power=10,
                     flow_v=3600, flow_m=3600, **units)
    p_ = converted['p']
    T_ = converted['T']
    speed_ = converted['speed']
    flow_m_ = converted['flow_m']
    flow_v_ = converted['flow_v']
    power_ = converted['power']

    assert_allclose(p_, 1698999.99999)
    assert_allclose(T_, 303.15)
    assert_allclose(speed_, 104.71975511965977)
    assert_allclose(flow_m_, 1)
    assert_allclose(flow_v_, 1)
    assert_allclose(power_, 10000)

    # wrong units
    with pytest.raises(ValueError):
        func(p=1, flow_m_units='kg**/h')


@pytest.fixture
def pure_fluid():
    return State.define(p=101008, T=273, fluid='CarbonDioxide')


def test_pure_fluid(pure_fluid):
    assert_allclose(pure_fluid.rhomass(), 1.9716931060214515)


@pytest.fixture
def state_si_air():
    fluid = {'Oxygen': 0.2096, 'Nitrogen': 0.7812, 'Argon': 0.0092}
    p = 101008
    T = 273
    return State.define(p=p, T=T, fluid=fluid, EOS='REFPROP')


def test_state_si_air(state_si_air):
    p = 101008
    T = 273
    rho = 1.2893942613777385
    k = 1.4027565065533025
    assert_allclose(state_si_air.p(), p, rtol=1e-4)
    assert_allclose(state_si_air.T(), T, rtol=1e-4)
    assert_allclose(state_si_air.rhomass(), rho, rtol=1e-4)
    assert_allclose(state_si_air.k(), k, rtol=1e-4)


@pytest.fixture
def state_en_air():
    fluid = {'Oxygen': 0.2096, 'Nitrogen': 0.7812, 'Argon': 0.0092}
    p = 14.649971812683193
    T = 31.73000040000001
    units = {'p_units': 'psi', 'T_units': 'degF'}
    return State.define(fluid=fluid, p=p, T=T, EOS='REFPROP', **units)


def test_state_en_air(state_en_air):
    p = 101008
    T = 273
    rho = 1.289394261380401
    assert_allclose(state_en_air.p(), p, rtol=1e-4)
    assert_allclose(state_en_air.T(), T, rtol=1e-4)
    assert_allclose(state_en_air.rhomass(), rho, rtol=1e-4)


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

    return State.define(p=16.99, T=38.4, fluid=fluid, EOS='REFPROP', **units)


def test_state_si_main_op(state_si_main_op):
    p = 1699000
    T = 311.5499999
    rhomass = 16.176361737467335
    assert_allclose(state_si_main_op.p(), p, rtol=1e-4)
    assert_allclose(state_si_main_op.T(), T, rtol=1e-4)
    assert_allclose(state_si_main_op.rhomass(), rhomass, rtol=1e-4)


@pytest.fixture
def state_si_main_test():
    fluid = {'CarbonDioxide': 0.76064,
             'R134a': 0.23581,
             'Nitrogen': 0.00284,
             'Oxygen': 0.00071}
    units = {'p_units': 'bar', 'T_units': 'degK'}

    return State.define(p=1.839, T=291.5, fluid=fluid, EOS='REFPROP', **units)


def test_state_si_main_test(state_si_main_test):
    p = 183900
    T = 291.5
    rhomass = 4.436768406942847
    assert_allclose(state_si_main_test.p(), p, rtol=1e-4)
    assert_allclose(state_si_main_test.T(), T, rtol=1e-4)
    assert_allclose(state_si_main_test.rhomass(), rhomass, rtol=1e-4)


def test_copy(state_si_main_test):
    p = 183900
    T = 291.5
    rhomass = 4.436768406942847
    assert_allclose(state_si_main_test.p(), p, rtol=1e-4)
    assert_allclose(state_si_main_test.T(), T, rtol=1e-4)
    assert_allclose(state_si_main_test.rhomass(), rhomass, rtol=1e-4)

    s1 = copy(state_si_main_test)
    s2 = copy(state_si_main_test)

    s2.update(CP.PT_INPUTS, 200000, 300)
    assert_allclose(s2.p(), 200000, rtol=1e-4)
    assert_allclose(s2.T(), 300, rtol=1e-4)
    assert_allclose(s2.rhomass(), 4.687447306413212, rtol=1e-4)
    assert state_si_main_test != s2

    assert_allclose(s1.p(), p, rtol=1e-4)
    assert_allclose(s1.T(), T, rtol=1e-4)
    assert_allclose(s1.rhomass(), rhomass, rtol=1e-4)
    assert state_si_main_test != s1


def test_heos_error():
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

    with pytest.raises(ValueError):
        State.define(p=16.99, T=38.4, fluid=fluid, EOS='HEOS', **units)


@pytest.mark.skipif(skip is True, reason='Slow test')
def test_ps_hs_ds():
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

    ps = State.define(p=16.99, s=4163.202483953783, fluid=fluid, **units)
    assert_allclose(ps.T(), 311.54999999999995, rtol=1e-4)

    ds = State.define(d=16.176361737467335, s=4163.202483953783, fluid=fluid, **units)
    assert_allclose(ds.T(), 311.54999999999995, rtol=1e-4)

    # hs may cause an error when all tests are carried out at the same time
    # this does not occur if we only run test_state
    hs = State.define(h=741544.2914857446, s=4163.202483953783, fluid=fluid, **units)
    assert_allclose(hs.T(), 311.54999999999995, rtol=1e-4)



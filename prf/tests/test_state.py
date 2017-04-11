import pytest
import CoolProp.CoolProp as CP
from prf.state import *
from copy import copy
from numpy.testing import assert_allclose


def test_convert_units():
    p = 1698999.99999
    T = 303.15
    speed = 104.71975511965977

    units = {'T_units': 'degC', 'p_units': 'bar', 'speed_units': 'RPM'}

    @convert_to_base_units
    def func(**kwargs):
        return kwargs

    converted = func(p=16.99, T=30, speed=1000, **units)
    p_ = converted['p']
    T_ = converted['T']
    speed_ = converted['speed']

    assert_allclose(p_, p)
    assert_allclose(T_, T)
    assert_allclose(speed_, speed)


@pytest.fixture
def state_si_air():
    fluid = {'Oxygen': 0.2096, 'Nitrogen': 0.7812, 'Argon': 0.0092}
    p = 101008
    T = 273
    return State.define(p=p, T=T, fluid=fluid, EOS='HEOS')


def test_state_si_air(state_si_air):
    p = 101008
    T = 273
    rho = 1.2893965217814896
    assert_allclose(state_si_air.p(), p)
    assert_allclose(state_si_air.T(), T)
    assert_allclose(state_si_air.rhomass(), rho)


@pytest.fixture
def state_en_air():
    fluid = {'Oxygen': 0.2096, 'Nitrogen': 0.7812, 'Argon': 0.0092}
    p = 14.649971812683193
    T = 31.73000040000001
    units = {'p_units': 'psi', 'T_units': 'degF'}
    return State.define(fluid=fluid, p=p, T=T, EOS='HEOS', **units)


def test_state_en_air(state_en_air):
    p = 101008
    T = 273
    rho = 1.2893965217814896
    assert_allclose(state_en_air.p(), p)
    assert_allclose(state_en_air.T(), T)
    assert_allclose(state_en_air.rhomass(), rho)


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

    return State.define(p=16.99, T=38.4, fluid=fluid, EOS='HEOS', **units)


def test_state_si_main_op(state_si_main_op):
    p = 1699000
    T = 311.5499999
    rhomass = 16.176449459148156
    assert_allclose(state_si_main_op.p(), p)
    assert_allclose(state_si_main_op.T(), T)
    assert_allclose(state_si_main_op.rhomass(), rhomass)


@pytest.fixture
def state_si_main_test():
    fluid = {'CarbonDioxide': 0.76064,
             'R134a': 0.23581,
             'Nitrogen': 0.00284,
             'Oxygen': 0.00071}
    units = {'p_units': 'bar', 'T_units': 'degK'}

    return State.define(p=1.839, T=291.5, fluid=fluid, EOS='HEOS', **units)


def test_state_si_main_test(state_si_main_test):
    p = 183900
    T = 291.5
    rhomass = 4.436646748577415
    assert_allclose(state_si_main_test.p(), p)
    assert_allclose(state_si_main_test.T(), T)
    assert_allclose(state_si_main_test.rhomass(), rhomass)


def test_copy(state_si_main_test):
    p = 183900
    T = 291.5
    rhomass = 4.436646748577415
    assert_allclose(state_si_main_test.p(), p)
    assert_allclose(state_si_main_test.T(), T)
    assert_allclose(state_si_main_test.rhomass(), rhomass)

    s1 = copy(state_si_main_test)
    s2 = copy(state_si_main_test)

    s2.update(CP.PT_INPUTS, 200000, 300)
    assert_allclose(s2.p(), 200000)
    assert_allclose(s2.T(), 300)
    assert_allclose(s2.rhomass(), 4.687447306413212, rtol=1e-4)
    assert state_si_main_test != s2

    assert_allclose(s1.p(), p)
    assert_allclose(s1.T(), T)
    assert_allclose(s1.rhomass(), rhomass, rtol=1e-4)
    assert state_si_main_test != s1


@pytest.fixture
def state_si_main_op_REFPROP():
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


def test_state_si_main_op_REFPROP(state_si_main_op_REFPROP):
    p = 1699000
    T = 311.5499999
    rhomass = 16.176330381770875
    assert_allclose(state_si_main_op_REFPROP.p(), p)
    assert_allclose(state_si_main_op_REFPROP.T(), T)
    assert_allclose(state_si_main_op_REFPROP.rhomass(), rhomass, rtol=1e-4)

# TODO add test to copy method
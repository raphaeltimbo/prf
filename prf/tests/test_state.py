import pytest
from prf.state import *
import numpy as np
from numpy.testing import assert_allclose


@pytest.fixture
def state_si_air():
    fluid = {'Oxygen': 0.2096, 'Nitrogen': 0.7812, 'Argon': 0.0092}
    p = 101008
    T = 273
    return State.define('HEOS', fluid, p, T)


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
    return State.define('HEOS', fluid, p, T, **units)


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

    return State.define('HEOS', fluid, 16.99, 38.4, **units)


def test_state_si_main_op(state_si_main_op):
    p = 1699000
    T = 311.5499999
    rhomass = 16.162687790285435
    assert_allclose(state_si_main_op.p(), p)
    assert_allclose(state_si_main_op.T(), T)
    assert_allclose(state_si_main_op.rhomass(), rhomass)

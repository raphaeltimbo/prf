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
    assert_allclose(state_si_air.p(), 101008)
    assert_allclose(state_si_air.T(), 273)
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
    assert_allclose(state_en_air.p(), 101008)
    assert_allclose(state_en_air.T(), 273)
    assert_allclose(state_en_air.rhomass(), rho)
import pytest
from prf.state import *
from prf.curves import *
from numpy.testing import assert_allclose


@pytest.fixture
def suc_1():
    fluid = {'CarbonDioxide': 0.76064,
             'R134a': 0.23581,
             'Nitrogen': 0.00284,
             'Oxygen': 0.00071}
    units = {'p_units': 'bar', 'T_units': 'degK'}

    return State.define(fluid, 1.839, 291.5, EOS='HEOS', **units)


@pytest.fixture
def disch_1():
    fluid = {'CarbonDioxide': 0.76064,
             'R134a': 0.23581,
             'Nitrogen': 0.00284,
             'Oxygen': 0.00071}
    units = {'p_units': 'bar', 'T_units': 'degK'}

    return State.define(fluid, 5.902, 380.7, EOS='HEOS', **units)


def test_n_exp(suc_1, disch_1):
    assert_allclose(n_exp(suc_1, disch_1), 1.2910807257829124)


def test_head_pol(suc_1, disch_1):
    assert_allclose(head_pol(suc_1, disch_1), 55282.59221757925)


def test_ef_pol(suc_1, disch_1):
    assert_allclose(ef_pol(suc_1, disch_1), 0.7111884556585154)

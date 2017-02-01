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
    suc = State.define(fluid, 1.839, 291.5, **units)
    disch = State.define(fluid, 5.902, 380.7, **units)

    curve1 = np.array([[7941.00, 7964.00],
                       [34203.60, 26542.80],
                       [2.238, 1.873],
                       [298.3, 297.7],
                       [7.255, 7.44],
                       [391.1, 399.6]])
    curve = Curve.from_discharge(fluid, curve1, **units)
    b = 0.0285
    D = 0.365
    return Impeller([curve], b, D)


def test_impeller1(impeller1):
    non_dim_curves_0 = np.array([[  3.26149996e-04,   3.01619832e-04],
                                 [  2.45924494e+02,   2.90347051e+02],
                                 [  7.58596759e-01,   8.16683236e-01]])
    assert_allclose(impeller1.non_dim_curves[0], non_dim_curves_0)


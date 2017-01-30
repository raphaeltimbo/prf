import numpy as np
import CoolProp as CP
from copy import copy
from prf.state import *


__all__ = ['Curves', 'n_exp', 'head_pol', 'ef_pol', 'head_isen', 'ef_isen',
           'schultz_f']


class Curves:
    def __init__(self, curves, ps, Ts, fluid):
        """
        Construct curves given a speed and an array with
        flow, head and efficiency.
        Parameters
        ----------
        curves : array
            Array with the curves as:
            array([speed],          -> RPM
                  [flow_m],         -> kg/h
                  [head],           -> J/kg
                  [efficiency])     -> %
        ps : float
            Suction pressure.
        Ts : float
            Suction temperature.
        fluid : dict
            Dictionary with constituent and composition
            (e.g.: ({'Oxygen': 0.2096, 'Nitrogen': 0.7812, 'Argon': 0.0092})

        Returns
        -------

        Attributes
        ----------

        Examples
        --------

        """
        self.curves = curves
        self.ps = ps
        self.Ts = Ts
        self.fluid = fluid

        # construct state
        self.suc_state = State.define('HEOS', self.flow_m, self.ps, self.Ts)
        self.speed = curves[0]
        self.flow_m = curves[1]
        self.head = curves[2]
        self.efficiency = curves[3]

    @classmethod
    def from_discharge(cls, curves, ps, Ts, fluid, **kwargs):
        """
                Construct curves given a speed and an array with
        flow, head and efficiency.
        Parameters
        ----------
        curves : array
            Array with the curves as:
            array([speed],          -> RPM
                  [flow_m],         -> kg/h
                  [pd],             -> Pa
                  [Td])             -> K
        ps : float
            Suction pressure.
        Ts : float
            Suction temperature.
        fluid : dict
            Dictionary with constituent and composition
            (e.g.: ({'Oxygen': 0.2096, 'Nitrogen': 0.7812, 'Argon': 0.0092})

        Returns
        -------

        Attributes
        ----------

        Examples
        --------
        """
        # suction state
        # suc = State('HEOS', ps, Ts, fluid, **kwargs)

        # calculate head and efficiency for each point

    # TODO add **kwargs for units
    # TODO add constructor -> from_discharge_conditions


def n_exp(suc, disch):
    """Polytropic exponent.

    Calculates the polytropic exponent given a suction and a discharge state.

    Parameters:
    -----------
    suc : State
        Suction state.
    disch : State
        Discharge state.

    Returns:
    --------
    n_exp : float
        Polytropic exponent.

    Examples:
    ---------

    """
    ps = suc.p()
    vs = 1 / suc.rhomass()
    pd = disch.p()
    vd = 1 / disch.rhomass()

    return np.log(pd/ps)/np.log(vs/vd)


def head_pol(suc, disch):
    """Polytropic head.

    Calculates the polytropic head given a suction and a discharge state.

    Parameters:
    -----------
    suc : State
        Suction state.
    disch : State
        Discharge state.

    Returns:
    --------
    head_pol : float
        Polytropic head.

    Examples:
    ---------

    """
    n = n_exp(suc, disch)

    p2 = disch.p()
    v2 = 1 / disch.rhomass()
    p1 = suc.p()
    v1 = 1 / suc.rhomass()

    return (n/(n-1))*(p2*v2 - p1*v1)


def ef_pol(suc, disch):
    """Polytropic efficiency.

    Calculates the polytropic efficiency given suction and discharge state.

    Parameters:
    -----------
    suc : State
        Suction state.
    disch : State
        Discharge state.

    Returns:
    --------
    ef_pol : float
        Polytropic head.

    Examples:
    ---------

    """
    wp = head_pol(suc, disch)
    dh = disch.hmass() - suc.hmass()
    return wp/dh


def head_isen(suc, disch):
    """Isentropic head.

    Calculates the Isentropic head given a suction and a discharge state.

    Parameters:
    -----------
    suc : State
        Suction state.
    disch : State
        Discharge state.

    Returns:
    --------
    head_isen : float
        Isentropic head.

    Examples:
    ---------
    >>> fluid ={'CarbonDioxide': 0.76064,
    ...         'R134a': 0.23581,
    ...         'Nitrogen': 0.00284,
    ...         'Oxygen': 0.00071}
    >>> suc = State.define(fluid, 183900, 291.5)
    >>> disch = State.define(fluid, 590200, 380.7)
    >>> head_isen(suc, disch) # doctest: +ELLIPSIS
    53166.296...
    """
    # define state to isentropic discharge
    disch_s = copy(disch)
    disch_s.update(CP.PSmass_INPUTS, disch.p(), suc.smass())

    return head_pol(suc, disch_s)


def ef_isen(suc, disch):
    """Isentropic efficiency.

    Calculates the Isentropic efficiency given a suction and a discharge state.

    Parameters:
    -----------
    suc : State
        Suction state.
    disch : State
        Discharge state.

    Returns:
    --------
    ef_isen : float
        Isentropic efficiency.

    Examples:
    ---------
    >>> fluid ={'CarbonDioxide': 0.76064,
    ...         'R134a': 0.23581,
    ...         'Nitrogen': 0.00284,
    ...         'Oxygen': 0.00071}
    >>> suc = State.define(fluid, 183900, 291.5)
    >>> disch = State.define(fluid, 590200, 380.7)
    >>> ef_isen(suc, disch) # doctest: +ELLIPSIS
    0.684...
    """
    ws = head_isen(suc, disch)
    dh = disch.hmass() - suc.hmass()
    return ws/dh


def schultz_f(suc, disch):
    """Schultz factor.

    Calculates the Schultz factor given a suction and discharge state.
    This factor is used to correct the polytropic head as per PTC 10.

    Parameters:
    -----------
    suc : State
        Suction state.
    disch : State
        Discharge state.

    Returns:
    --------
    ef_isen : float
        Isentropic efficiency.

    Examples:
    ---------
    >>> fluid ={'CarbonDioxide': 0.76064,
    ...         'R134a': 0.23581,
    ...         'Nitrogen': 0.00284,
    ...         'Oxygen': 0.00071}
    >>> suc = State.define(fluid, 183900, 291.5)
    >>> disch = State.define(fluid, 590200, 380.7)
    >>> schultz_f(suc, disch) # doctest: +ELLIPSIS
    1.001...
    """
    # define state to isentropic discharge
    disch_s = copy(disch)
    disch_s.update(CP.PSmass_INPUTS, disch.p(), suc.smass())

    h2s_h1 = disch_s.hmass() - suc.hmass()
    h_isen = head_isen(suc, disch)

    return h2s_h1/h_isen


def head_pol_schultz(suc, disch):
    """Polytropic head corrected by the Schultz factor.

    Calculates the polytropic head corrected by the Schultz factor
    given a suction and a discharge state.

    Parameters:
    -----------
    suc : State
        Suction state.
    disch : State
        Discharge state.

    Returns:
    --------
    head_pol_schultz : float
        Polytropic head corrected by the Schultz factor.

    Examples:
    ---------
    >>> fluid ={'CarbonDioxide': 0.76064,
    ...         'R134a': 0.23581,
    ...         'Nitrogen': 0.00284,
    ...         'Oxygen': 0.00071}
    >>> suc = State.define(fluid, 183900, 291.5)
    >>> disch = State.define(fluid, 590200, 380.7)
    >>> head_pol_schultz(suc, disch) # doctest: +ELLIPSIS
    55377.434...
    """
    f = schultz_f(suc, disch)
    head = head_pol(suc, disch)

    return f*head

# TODO add schultz_factor
# TODO add head_pol_schultz

import numpy as np
import CoolProp as CP
from copy import copy
from scipy.optimize import newton
from prf.state import *


__all__ = ['Point', 'n_exp', 'head_pol', 'eff_pol', 'head_isen',
           'eff_isen', 'schultz_f', 'head_pol_schultz', 'eff_pol_schultz',
           'convert_to_base_units']


class Point:
    @convert_to_base_units
    def __init__(self, *args, **kwargs):
        """Point.

        A point in the compressor map that can be defined in different ways.

        Parameters
        ----------
        suc, disch : prf.State, prf.State
            Suction and discharge states for the point.

        suc, head, eff : prf.State, float, float
            Suction state, polytropic head and polytropic efficiency.

        suc, head, power : prf.State, float, float
            Suction state, polytropic head and gas power.

        suc, eff, vol_ratio : prf.State, float, float
            Suction state, polytropic efficiecy and volume ratio.

        Returns
        -------
        Point : prf.Point
            A point in the compressor map.


        """
        # TODO create dictionary with optional inputs
        self.suc = kwargs.get('suc')

        try:
            self.speed = kwargs['speed']
            if 'flow_m' not in kwargs:
                self.flow_v = kwargs['flow_v']
                self.flow_m = self.flow_v * self.suc.rhomass()
            else:
                self.flow_m = kwargs['flow_m']
                self.flow_v = self.flow_m / self.suc.rhomass()
        except KeyError as err:
            raise Exception('Argument not provided', err.args[0]) from err

        self.disch = kwargs.get('disch')
        self.head = kwargs.get('head')
        self.eff = kwargs.get('eff')
        self.power = kwargs.get('power')
        self.volume_ratio = kwargs.get('volume_ratio')

        if self.suc and self.disch is not None:
            self.calc_from_suc_disch(self.suc, self.disch)
        elif self.suc and self.head and self.eff is not None:
            self.calc_from_suc_head_eff(self.suc, self.head, self.eff)
        elif self.suc and self.head and self.power is not None:
            self.calc_from_suc_head_power(self.suc, self.head, self.power)
        elif self.suc and self.eff and self.volume_ratio is not None:
            self.calc_from_suc_eff_vol_ratio(self.suc, self.eff, self.volume_ratio)
        else:
            raise KeyError('Argument not provided')

        if self.volume_ratio is None:
            self.volume_ratio = self.suc.rhomass() / self.disch.rhomass()

        self.mach_comparison = kwargs.get('mach_comparison')
        self.reynolds_comparison = kwargs.get('reynolds_comparison')
        self.volume_ratio_comparison = kwargs.get('volume_ratio_comparison')

    def __repr__(self):
        return (
            '\nPoint: '
            + '\n Volume flow: {:10.5} m^3 / s'.format(self.flow_v)
            + '\n Head       : {:10.5} J / kg.K'.format(self.head)
            + '\n Efficiency : {:10.5} %'.format(100 * self.eff)
        )

    # TODO Put pol. head/eff and isen. head/eff functions inside point class

    def calc_from_suc_disch(self, suc, disch):
        self.head = head_pol_schultz(suc, disch)
        self.eff = eff_pol_schultz(suc, disch)
        self.power = power(self.flow_m, self.head, self.eff)

    def calc_from_suc_head_eff(self, suc, head, eff):
        """Point from suction, head and efficiency.

        This function will construct a point given its suction, head and
        efficiency. Discharge state is calculated by an iterative process
        where the discharge pressure is initially defined based on an
        isentropic compression. After defining the pressure, polytropic
        head is calculated and compared with the given head. A new pressure
        is defined and the process is repeated.

        Parameters
        ----------
        suc : state
            Suction state.
        head : float
            Polytropic head.
        eff : float
            Polytropic efficiency.

        Returns
        -------

        """
        # calculate discharge state from suction, head and efficiency
        h_suc = suc.hmass()
        h_disch = head/eff + h_suc

        # first disch state will consider an isentropic compression
        s_disch = suc.smass()
        disch = State.define(fluid=suc.fluid_dict(), h=h_disch, s=s_disch)

        def update_pressure(p):
            disch.update(CP.HmassP_INPUTS, h_disch, p)
            new_head = head_pol_schultz(suc, disch)

            return new_head - head

        newton(update_pressure, disch.p())

        self.disch = disch
        self.calc_from_suc_disch(suc, disch)

    def calc_from_suc_head_power(self, suc, head, power):
        # calculate efficiency
        self.eff = self.flow_m * head / power
        self.calc_from_suc_head_eff(suc, head, self.eff)

    def calc_from_suc_eff_vol_ratio(self, suc, eff, volume_ratio):
        # from volume ratio calculate discharge rhomass
        d_disch = suc.rhomass() / volume_ratio

        # first disch state will consider an isentropic compression
        s_disch = suc.smass()
        disch = State.define(fluid=suc.fluid_dict(), d=d_disch, s=s_disch)

        def update_pressure(p):
            disch.update(CP.DmassP_INPUTS, disch.rhomass(), p)
            new_eff = eff_pol_schultz(suc, disch)

            return new_eff - eff

        newton(update_pressure, disch.p())

        self.disch = disch
        self.calc_from_suc_disch(suc, disch)


def n_exp(suc, disch):
    """Polytropic exponent.

    Calculates the polytropic exponent given a suction and a discharge state.

    Parameters
    ----------
    suc : State
        Suction state.
    disch : State
        Discharge state.

    Returns
    -------
    n_exp : float
        Polytropic exponent.

    Examples
    --------

    """
    ps = suc.p()
    vs = 1 / suc.rhomass()
    pd = disch.p()
    vd = 1 / disch.rhomass()

    return np.log(pd/ps)/np.log(vs/vd)


def head_pol(suc, disch):
    """Polytropic head.

    Calculates the polytropic head given a suction and a discharge state.

    Parameters
    ----------
    suc : State
        Suction state.
    disch : State
        Discharge state.

    Returns
    -------
    head_pol : float
        Polytropic head.

    Examples
    --------

    """
    n = n_exp(suc, disch)

    p2 = disch.p()
    v2 = 1 / disch.rhomass()
    p1 = suc.p()
    v1 = 1 / suc.rhomass()

    return (n/(n-1))*(p2*v2 - p1*v1)


def eff_pol(suc, disch):
    """Polytropic efficiency.

    Calculates the polytropic efficiency given suction and discharge state.

    Parameters
    ----------
    suc : State
        Suction state.
    disch : State
        Discharge state.

    Returns
    -------
    eff_pol : float
        Polytropic head.

    Examples
    --------

    """
    wp = head_pol(suc, disch)
    dh = disch.hmass() - suc.hmass()
    return wp/dh


def head_isen(suc, disch):
    """Isentropic head.

    Calculates the Isentropic head given a suction and a discharge state.

    Parameters
    ----------
    suc : State
        Suction state.
    disch : State
        Discharge state.

    Returns
    -------
    head_isen : float
        Isentropic head.

    Examples
    --------
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


def eff_isen(suc, disch):
    """Isentropic efficiency.

    Calculates the Isentropic efficiency given a suction and a discharge state.

    Parameters
    ----------
    suc : State
        Suction state.
    disch : State
        Discharge state.

    Returns
    -------
    ef_isen : float
        Isentropic efficiency.

    Examples
    --------
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

    Parameters
    ----------
    suc : State
        Suction state.
    disch : State
        Discharge state.

    Returns
    -------
    ef_isen : float
        Isentropic efficiency.

    Examples
    --------
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

    Parameters
    ----------
    suc : State
        Suction state.
    disch : State
        Discharge state.

    Returns
    -------
    head_pol_schultz : float
        Polytropic head corrected by the Schultz factor.

    Examples
    --------
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

    return f * head


def eff_pol_schultz(suc, disch):
    wp = head_pol_schultz(suc, disch)
    dh = disch.hmass() - suc.hmass()
    return wp/dh


def power(flow_m, head, eff):
    """Power.

    Calculate the power consumption.

    Parameters
    ----------
    flow_m : float
        Mass flow.
    head : float
        Head.
    eff : float
        Polytropic efficiency.

    Returns
    -------
    power : float

    """
    return flow_m * head / eff

# TODO add head Mallen
# TODO add head Huntington
# TODO add head reference
# TODO add power

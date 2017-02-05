import numpy as np
import CoolProp as CP
from collections import namedtuple
from copy import copy
from scipy.optimize import newton
from prf.state import *


__all__ = ['Point', 'Curve', 'n_exp', 'head_pol', 'eff_pol', 'head_isen',
           'eff_isen', 'schultz_f', 'head_pol_schultz', 'convert_to_base_units']


class Point:
    @convert_to_base_units
    def __init__(self, *args, **kwargs):
        # TODO give options to mass or volume flow
        try:
            self.speed = kwargs['speed']
            self.flow_m = kwargs['flow_m']
        except KeyError as err:
            raise Exception('Argument not provided', err.args[0]) from err
        self.suc = kwargs.get('suc')
        self.disch = kwargs.get('disch')
        self.head = kwargs.get('head')
        self.eff = kwargs.get('eff')
        self.mach_diff = kwargs.get('mach_diff')

        if self.disch is None:
            self.calc_from_suc_head_eff(self.suc, self.head, self.eff)
        if self.suc and self.disch is not None:
            self.calc_from_suc_disch(self.suc, self.disch)

    def calc_from_suc_disch(self, suc, disch):
        self.head = head_pol_schultz(suc, disch)
        self.eff = eff_pol(suc, disch)

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

        # first disch state will consider and isentropic compression
        s_disch = suc.smass()
        disch = State.define(fluid=suc.fluid_dict(), h=h_disch, s=s_disch)

        def update_pressure(p):
            disch.update(CP.PT_INPUTS, p, disch.T())
            new_head = head_pol_schultz(suc, disch)

            return new_head - head

        newton(update_pressure, disch.p())

        self.disch = disch


class Curve:
    def __init__(self, fluid, curve):
        """
        Construct curve given a speed and an array with
        flow, head and efficiency.
        Parameters
        ----------
        fluid : dict
            Dictionary with constituent and composition
            (e.g.: ({'Oxygen': 0.2096, 'Nitrogen': 0.7812, 'Argon': 0.0092})
        ps : float
            Suction pressure.
        Ts : float
            Suction temperature.
        curve : array
            Array with the curve as:
            array([speed],          -> RPM
                  [flow_m],         -> kg/s
                  [ps],             -> Pa
                  [Ts],             -> K
                  [pd],             -> Pa
                  [Td],             -> K
                  [head],           -> J/kg
                  [efficiency])

        Returns
        -------

        Attributes
        ----------

        Examples
        --------

        """
        self.curve = curve
        self.fluid = fluid
        self.speed = curve[0]
        self.flow_m = curve[1]
        self.ps = curve[2]
        self.Ts = curve[3]
        self.pd = curve[4]
        self.Td = curve[5]
        self.head = curve[6]
        self.efficiency = curve[7]
        self.suc = curve[8]
        self.disch = curve[9]
        self.n = curve[10]

    def points(self):
        point = namedtuple('point', ['speed',
                                     'flow_m',
                                     'ps',
                                     'Ts',
                                     'pd',
                                     'Td',
                                     'head',
                                     'efficiency',
                                     'suc',
                                     'disch',
                                     'n'])

        for i in range(len(self.curve.T)):
            yield i, point(speed=self.speed[i],
                           flow_m=self.flow_m[i],
                           ps=self.ps[i],
                           Ts=self.Ts[i],
                           pd=self.pd[i],
                           Td=self.Td[i],
                           head=self.head[i],
                           efficiency=self.efficiency[i],
                           suc=self.suc[i],
                           disch=self.disch[i],
                           n=self.n[i])

    @classmethod
    @convert_to_base_units
    def from_discharge(cls, **kwargs):
        """
        Construct curve given a speed and an array with
        flow, suction and discharge conditions.

        Parameters
        ----------
        fluid : dict
            Dictionary with constituent and composition
            (e.g.: ({'Oxygen': 0.2096, 'Nitrogen': 0.7812, 'Argon': 0.0092})
        curve : array
            Array with the curve as:
            array([speed],          -> RPM
                  [flow_m],         -> kg/s
                  [ps],             -> Pa
                  [Ts],             -> K
                  [pd],             -> Pa
                  [Td])             -> K

        Returns
        -------

        Attributes
        ----------

        Examples
        --------
        """
        # TODO change array to construct curves from suc and disch states.
        fluid = kwargs.get('fluid')
        curve = kwargs.get('curve')

        curve_head = np.zeros([11, len(curve.T)], dtype=object)
        curve_head[:6] = curve[:6]

        # calculate head and efficiency for each point
        i = 0
        for point, point_new in zip(curve.T, curve_head.T):
            ps = point[2]
            Ts = point[3]
            suc = State.define(fluid=fluid, p=ps, T=Ts, **kwargs)

            pd = point[4]
            Td = point[5]
            disch = State.define(fluid=fluid, p=pd, T=Td, **kwargs)

            point_new[6] = head_pol_schultz(suc, disch)
            point_new[7] = ef_pol(suc, disch)
            point_new[8] = suc
            point_new[9] = disch
            point_new[10] = i
            i += 1

        return cls(fluid, curve_head)

    @classmethod
    def from_head_eff(cls):
        """
        Construct curve given a speed and an array with
        flow, head and efficiency.

        Parameters
        ----------
        fluid : dict
            Dictionary with constituent and composition
            (e.g.: ({'Oxygen': 0.2096, 'Nitrogen': 0.7812, 'Argon': 0.0092})
        curve : array
            Array with the curve as:
            array([speed],          -> rad/s
                  [flow_m],         -> kg/s
                  [suc],
                  [head],           -> J/kg
                  [efficiency])

        Returns
        -------

        Attributes
        ----------

        Examples
        --------
        """
        pass

    # TODO add **kwargs for units
    # TODO add constructor -> from_discharge_conditions


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
    ef_pol : float
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

    return f*head

# TODO add head Mallen
# TODO add head Huntington
# TODO add head reference
# TODO add power

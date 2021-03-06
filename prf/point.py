import numpy as np
import pandas as pd
import CoolProp as CP
import matplotlib.pyplot as plt
import warnings
from copy import copy
from scipy.optimize import newton
from prf.state import *


__all__ = ['Point', 'Curve', 'convert_to_base_units']


class Point:
    """Point.

    A point in the compressor map that can be defined in different ways.

    Parameters
    ----------
    speed : float
        Speed in 1/s.
    flow_v or flow_m : float
        Volumetric or mass flow.
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
    @convert_to_base_units
    def __init__(self, *args, **kwargs):

        # TODO create dictionary with optional inputs
        self.suc = kwargs.get('suc')
        # dummy state used to avoid copying states
        self._dummy_state = copy(self.suc)

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
            + '\n Power      : {:10.5} W'.format(self.power)
        )

    def calc_from_suc_disch(self, suc, disch):
        self.head = self.head_pol_schultz()
        self.eff = self.eff_pol_schultz()
        self.power = self.power_calc()

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
        if disch.not_defined():
            raise ValueError(f'state not defined: {disch}')

        def update_pressure(p):
            disch.update(CP.HmassP_INPUTS, h_disch, p)
            new_head = self.head_pol_schultz(suc, disch)

            return new_head - head

        # with newton we have a risk of falling outside a reasonable state
        # region. #TODO evaluate pressure interval to use brent method

        newton(update_pressure, disch.p(), tol=1e-4)

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
            new_eff = self.eff_pol_schultz(suc=suc, disch=disch)

            return new_eff - eff

        newton(update_pressure, disch.p(), tol=1e-4)

        self.disch = disch
        self.calc_from_suc_disch(suc, disch)

    def n_exp(self, suc=None, disch=None):
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
        if suc is None:
            suc = self.suc
        if disch is None:
            disch = self.disch

        ps = suc.p()
        vs = 1 / suc.rhomass()
        pd = disch.p()
        vd = 1 / disch.rhomass()

        return np.log(pd/ps)/np.log(vs/vd)

    def head_pol(self, suc=None, disch=None):
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
        if suc is None:
            suc = self.suc
        if disch is None:
            disch = self.disch

        n = self.n_exp(suc, disch)

        p2 = disch.p()
        v2 = 1 / disch.rhomass()
        p1 = suc.p()
        v1 = 1 / suc.rhomass()

        return (n/(n-1))*(p2*v2 - p1*v1)

    def eff_pol(self, suc=None, disch=None):
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
        if suc is None:
            suc = self.suc
        if disch is None:
            disch = self.disch

        wp = self.head_pol(suc, disch)

        dh = disch.hmass() - suc.hmass()

        return wp/dh

    def head_isen(self, suc=None, disch=None):
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
        if suc is None:
            suc = self.suc
        if disch is None:
            disch = self.disch

        # define state to isentropic discharge using dummy state
        disch_s = self._dummy_state
        disch_s.update(CP.PSmass_INPUTS, disch.p(), suc.smass())

        return self.head_pol(suc, disch_s)

    def eff_isen(self, suc=None, disch=None):
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
        if suc is None:
            suc = self.suc
        if disch is None:
            disch = self.disch

        ws = self.head_isen(suc, disch)
        dh = disch.hmass() - suc.hmass()
        return ws/dh

    def schultz_f(self, suc=None, disch=None):
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
        if suc is None:
            suc = self.suc
        if disch is None:
            disch = self.disch

        # define state to isentropic discharge using dummy state
        disch_s = self._dummy_state
        disch_s.update(CP.PSmass_INPUTS, disch.p(), suc.smass())

        h2s_h1 = disch_s.hmass() - suc.hmass()
        h_isen = self.head_isen(suc, disch)

        return h2s_h1/h_isen

    def head_pol_schultz(self, suc=None, disch=None):
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
        if suc is None:
            suc = self.suc
        if disch is None:
            disch = self.disch

        f = self.schultz_f(suc, disch)
        head = self.head_pol(suc, disch)

        return f * head

    def eff_pol_schultz(self, suc=None, disch=None):
        if suc is None:
            suc = self.suc
        if disch is None:
            disch = self.disch

        wp = self.head_pol_schultz(suc, disch)
        dh = disch.hmass() - suc.hmass()

        return wp/dh

    def power_calc(self, flow_m=None, head=None, eff=None):
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
        if flow_m is None:
            flow_m = self.flow_m
        if head is None:
            head = self.head
        if eff is None:
            eff = self.eff

        return flow_m * head / eff

# TODO add head Mallen
# TODO add head Huntington
# TODO add head reference
# TODO add power


class InterpolatedCurve(np.poly1d):
    """Auxiliary class to create interpolated curve.

    This class inherit from np.poly1d, changing the polydegree to a
    property to that its value can be changed after instantiation.

    Plots are also added in this class.
    """
    def __init__(self, x, y, deg=3, ylabel=None):
        """
        Auxiliary function to create an interpolated curve
        for each various points attributes.

        Parameters
        ----------
        x : array_like, shape (M,)
            x-coordinates of the M sample points (x[i], y[i]).
        y : array_like, shape (M,) or (M, K)
            y-coordinates of the sample points. Several data sets of sample points sharing the same x-coordinates can be fitted at once by passing in a 2D-array that contains one dataset per column.
        deg : int
            Degree of the fitting polynomial

        Returns
        -------
        interpolated_curve : np.poly1d
            Interpolated curve using np.poly1d function.
        """
        # create a list for the attribute iterating on the points list
        self.x = x
        self.y = y
        self._deg = deg
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            self.args = np.polyfit(self.x, self.y, self._deg)
        self.ylabel = ylabel

        super().__init__(self.args)

    @property
    def deg(self):
        return self._deg

    @deg.setter
    def deg(self, value):
        self._deg = value
        self.__init__(self.x, self.y, self._deg, self.ylabel)

    def plot(self, ax=None, plot_kws=None):
        if ax is None:
            ax = plt.gca()

        if plot_kws is None:
            plot_kws = dict()

        flow = np.linspace(self.x[0], self.x[-1], 20)
        ax.plot(flow, self(flow), **plot_kws)

        ax.set_xlabel('Volumetric flow $(m^3 / s)$')
        ax.set_ylabel(self.ylabel)
        return ax


class Curve:
    """Curve.
    
    A curve is a collection of points that share the same suction
    state and the same speed.
    
    Parameters
    ----------
    
    points : list
        List with the points
    
    """

    def __init__(self, points):
        # for one single point:
        if not isinstance(points, list):
            p0 = points
            p1 = Point(suc=p0.suc, eff=p0.eff, volume_ratio=p0.volume_ratio,
                       speed=p0.speed, flow_m=p0.flow_m+1)
            points = [p0, p1]

        self.points = sorted(points, key=lambda point: point.flow_v)

        # set each point as attribute
        for i, p in enumerate(self.points):
            setattr(self, 'point_' + str(i), p)

        # get one point to extract attributes
        self._point0 = self.points[0]
        self.suc = self._point0.suc
        self.speed = self._point0.speed

        # attributes from each point
        self.flow_v = [p.flow_v for p in self.points]
        self.flow_m = [p.flow_m for p in self.points]
        self.suc_p = [p.suc.p() for p in self.points]
        self.suc_T = [p.suc.T() for p in self.points]
        self.disch_p = [p.disch.p() for p in self.points]
        self.disch_T = [p.disch.T() for p in self.points]
        self.head = [p.head for p in self.points]
        self.eff = [p.eff for p in self.points]
        self.power = [p.power for p in self.points]

        # interpolated curves
        self.suc_p_curve = InterpolatedCurve(self.flow_v, self.suc_p,
                                             ylabel='Pressure $(Pa)$')
        self.suc_T_curve = InterpolatedCurve(self.flow_v, self.suc_T,
                                             ylabel='Temperature $(K)$')
        self.disch_p_curve = InterpolatedCurve(self.flow_v, self.disch_p,
                                               ylabel='Pressure $(Pa)$')
        self.disch_T_curve = InterpolatedCurve(self.flow_v, self.disch_T,
                                               ylabel='Temperature $(K)$')
        self.head_curve = InterpolatedCurve(self.flow_v, self.head,
                                            ylabel='Head $(J / kg)$')
        self.eff_curve = InterpolatedCurve(self.flow_v, self.eff,
                                           ylabel='Efficiency')
        self.power_curve = InterpolatedCurve(self.flow_v, self.power,
                                             ylabel='Power $(W)$')

    def __getitem__(self, item):
        return self.points[item]

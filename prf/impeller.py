import numpy as np
import CoolProp.CoolProp as CP
from .curve import *
from warnings import warn
from copy import copy


__all__ = ['Impeller', 'NonDimPoint']


class Impeller:
    def __init__(self, points, b, D, e=0.87e-6):
        """
        Impeller instance is initialized with the dimensional curve.
        The created instance will hold instances of the dimensional curves
        and of the non dimensional curves generated.

        Parameters
        ----------
        points : list
            List with points instances.
        b : float
            Impeller width (m).
        D : float
            Impeller diameter (m).
        e : float
            Impeller roughness.
            Defaults to 0.87 um.

        Returns
        -------
        non_dim_points : list
            List with non dimensional point instances.

        Attributes
        ----------

        Examples
        --------
        """
        # TODO define speed and suction state as properties and calculate current curve.
        # for one single point:
        if not isinstance(points, list):
            points = [points]
        self.points = points
        self.b = b
        self.D = D
        self.e = e
        self.non_dim_points = []
        for point in points:
            self.non_dim_points.append(NonDimPoint.from_impeller(self, point))

        # impeller current state
        self._suc = self.points[0].suc
        self._speed = self.points[0].speed
        self._flow_v = self.points[0].flow_v

        # the current points and curve
        self.new_points = None
        self.not_valid_points = None
        self.suc_p_curve = None
        self.suc_T_curve = None
        self.disch_p_curve = None
        self.disch_T_curve = None
        self.head_curve = None
        self.eff_curve = None
        self.power_curve = None
        self.current_point = None
        self.disch = None
        self._calc_new()

    @property
    def suc(self):
        return self._suc

    @suc.setter
    def suc(self, new_suc):
        self._suc = new_suc
        self._calc_new()

    @property
    def speed(self):
        return self._speed

    @speed.setter
    def speed(self, new_speed):
        self._speed = new_speed
        self._calc_new()

    @property
    def flow_v(self):
        return self._flow_v

    @flow_v.setter
    def flow_v(self, new_flow_v):
        self._flow_v = new_flow_v
        self._calc_new()

    def _calc_new(self):
        self.new_points = [self.new_point(self.suc, self.speed, i)
                           for i in range(len(self.points))]

        flow_v = [p.flow_v for p in self.new_points]
        suc_p = [p.suc.p() for p in self.new_points]
        suc_T = [p.suc.T() for p in self.new_points]
        disch_p = [p.disch.p() for p in self.new_points]
        disch_T = [p.disch.T() for p in self.new_points]
        head = [p.head for p in self.new_points]
        eff = [p.eff for p in self.new_points]
        power = [p.power for p in self.new_points]

        if len(flow_v) > 2:
            self.suc_p_curve = np.poly1d(np.polyfit(flow_v, suc_p, 3))
            self.suc_T_curve = np.poly1d(np.polyfit(flow_v, suc_T, 3))
            self.disch_p_curve = np.poly1d(np.polyfit(flow_v, disch_p, 3))
            self.disch_T_curve = np.poly1d(np.polyfit(flow_v, disch_T, 3))
            self.head_curve = np.poly1d(np.polyfit(flow_v, head, 3))
            self.eff_curve = np.poly1d(np.polyfit(flow_v, eff, 3))
            self.power_curve = np.poly1d(np.polyfit(flow_v, power, 3))

        current_disch_p = self.disch_p_curve(self.flow_v)
        current_disch_T = self.disch_T_curve(self.flow_v)
        current_disch = copy(self.suc)
        current_disch.update(CP.PT_INPUTS, current_disch_p, current_disch_T)

        self.disch = current_disch
        self.current_point = Point(suc=self.suc,
                                   disch=current_disch,
                                   flow_v=self.flow_v,
                                   speed=self.speed)
        self.check_similarity()

    def check_similarity(self):
        not_valid_points = {}

        for i, p in enumerate(self.new_points):
            if not all([p.mach_comparison['valid'],
                        p.reynolds_comparison['valid'],
                        p.volume_ratio_comparison['valid']]):
                not_valid_points['p' + str(i)] = p

        if len(not_valid_points) > 0:
            pts = ', '.join(not_valid_points)
            warn('Following points out of similarity: %s' % pts)

        self.not_valid_points = not_valid_points

    def flow_coeff(self, flow_m=None, suc=None, speed=None, point=None):
        """Flow coefficient.

        Calculates the flow coefficient for a point given the mass flow,
        suction state and speed.

        Parameters:
        -----------
        flow_m : float
            Mass flow (kg/s)
        suc : state
            Suction state.
        speed : float
            Speed in rad/s.

        Returns:
        --------
        flow_coeff : float
            Flow coefficient (non dimensional).

        Examples:
        ---------

        """
        if point is not None:
            if isinstance(point, int):
                point = self.points[point]
            flow_m = point.flow_m
            suc = point.suc
            speed = point.speed

        v = 1 / suc.rhomass()
        u = self.tip_speed(speed)

        # 3.2.5 ISO-5389
        flow_coeff = (flow_m * v * 4 /
                      (np.pi * self.D**2 * u))

        return flow_coeff

    def tip_speed(self, speed=None, point=None):
        """Impeller tip speed.

        Calculates the impeller tip speed for a given speed.

        Parameters
        ----------
        speed : float
            Speed in rad/s.

        Returns
        -------
        tip_speed : float
            Impeller tip speed in m/s.

        Examples
        --------

        """
        # TODO check dimensions and units
        if point is not None:
            if isinstance(point, int):
                point = self.points[point]
            speed = point.speed

        u = speed * self.D / 2

        return u

    def head_coeff(self, head=None, speed=None, point=None):
        """Head coefficient.

        Calculates the head coefficient given a head and speed.

        Parameters
        ----------
        head : float
            Head in J/kg.
        speed : float
            Speed in rad/s.

        Returns
        -------
        head_coeff : float
            Head coefficient (non dimensional).
        """
        if point is not None:
            if isinstance(point, int):
                point = self.points[point]

            head = point.head
            speed = point.speed

        u = self.tip_speed(speed)

        head_coeff = 2 * head / u**2  # 3.2.6 ISO-5389

        return head_coeff

    def mach(self, suc=None, speed=None, point=None):
        if point is not None:
            if isinstance(point, int):
                point = self.points[point]
            suc = point.suc
            speed = point.speed

        u = self.tip_speed(speed)
        a = suc.speed_sound()

        mach = u / a  # 3.2.3 ISO-5389

        return mach

    def reynolds(self, suc=None, speed=None, point=None):
        if point is not None:
            if isinstance(point, int):
                point = self.points[point]
            suc = point.suc
            speed = point.speed

        u = self.tip_speed(speed)
        b = self.b
        v = suc.viscosity() / suc.rhomass()  # E.115 ISO-5389

        reynolds = u * b / v  # 3.2.4 ISO-5389

        return reynolds

    def volume_ratio(self, suc=None, disch=None, point=None):
        if point is not None:
            if isinstance(point, int):
                point = self.points[point]
            suc = point.suc
            disch = point.disch

        volume_ratio = suc.rhomass() / disch.rhomass()

        return volume_ratio

    @convert_to_base_units
    def new_point(self, suc, speed, idx, **kwargs):
        """Curve.

        Calculates a new point based on the given suction state and speed.
        """
        # TODO check the closest flow. Add new arg point?
        # check points with flow within +- 5%
        # use mach to check the best non dim curve to be used
        # mach_new = self.mach(suc, speed)
        #
        # diff_mach = []
        # for point in self.points:
        #     mach_ = self.mach(point=point)
        #     diff_mach.append(mach_new - mach_)
        # idx = diff_mach.index(min(diff_mach))

        point_old = self.points[idx]
        non_dim_point = self.non_dim_points[idx]
        # store mach, reynolds and volume ratio from original point
        mach_old = self.mach(point=point_old)
        reynolds_old = self.reynolds(point=point_old)
        volume_ratio_old = self.volume_ratio(point=point_old)

        rho = suc.rhomass()
        u = self.tip_speed(speed)

        # calculate the mass flow
        phi = non_dim_point.flow_coeff
        flow_v = phi * np.pi * self.D**2 * u / 4
        flow_m = flow_v * rho

        # calculate new head and efficiency
        psi = non_dim_point.head_coeff
        head = psi * u**2 / 2
        eff = non_dim_point.eff

        point_new = Point(flow_m=flow_m, speed=speed, suc=suc, head=head, eff=eff)
        # store mach, reynolds and volume ratio from original point
        mach_new = self.mach(point=point_new)
        reynolds_new = self.reynolds(point=point_new)
        volume_ratio_new = self.volume_ratio(point=point_new)

        point_new.mach_comparison = compare_mach(mach_sp=mach_new,
                                                 mach_t=mach_old)
        point_new.reynolds_comparison = compare_reynolds(reynolds_sp=reynolds_new,
                                                         reynolds_t=reynolds_old)
        point_new.volume_ratio_comparison = compare_volume_ratio(ratio_sp=volume_ratio_new,
                                                                 ratio_t=volume_ratio_old)

        return point_new


class NonDimPoint:
    def __init__(self, *args, **kwargs):
        # calculate non dimensional curve
        self.flow_coeff = kwargs.get('flow_coeff')
        self.head_coeff = kwargs.get('head_coeff')
        self.eff = kwargs.get('eff')

    @classmethod
    def from_impeller(cls, impeller, point):
        # flow coefficient
        # calculate non dim curve and append
        flow_coeff = impeller.flow_coeff(point.flow_m,
                                         point.suc,
                                         point.speed)
        head_coeff = impeller.head_coeff(point.head,
                                         point.speed)
        eff = point.eff

        return cls(flow_coeff=flow_coeff, head_coeff=head_coeff, eff=eff)


def compare_mach(mach_sp, mach_t):
    """Compare mach numbers.

    Compares the mach numbers and evaluates
    them according to the PTC10 criteria.

    Parameters
    ----------
    mach_sp : float
        Mach number from specified condition.
    mach_t : float
        Mach number from test condition.

    Returns
    -------
    Dictionary with diff (Mmsp - Mmt), valid (True if diff is within limits),
    lower limit and upper limit.
    """
    if mach_sp < 0.214:
        lower_limit = -mach_sp
        upper_limit = -0.25 * mach_sp + 0.286
    elif 0.215 < mach_sp < 0.86:
        lower_limit = 0.266 * mach_sp - 0.271
        upper_limit = -0.25 * mach_sp + 0.286
    else:
        lower_limit = -0.042
        upper_limit = 0.07

    diff = mach_sp - mach_t

    if lower_limit < diff < upper_limit:
        valid = True
    else:
        valid = False

    return {'diff': diff, 'valid': valid,
            'lower_limit': lower_limit, 'upper_limit': upper_limit}


def compare_reynolds(reynolds_sp, reynolds_t):
    """Compare reynolds numbers.

    Compares the reynolds numbers and evaluates
    them according to the PCT10 criteria.

    Parameters
    ----------
    reynolds_sp : float
        Reynolds number from specified condition.
    reynolds_t : float
        Reynolds number from test condition.

    Returns
    -------
    Dictionary with ratio (Ret/Resp), valid (True if ratio is within limits),
    lower limit and upper limit.
    """
    x = (reynolds_sp/1e7)**0.3

    if 9e4 < reynolds_sp < 1e7:
        upper_limit = 100**x
    elif 1e7 < reynolds_sp:
        upper_limit = 100
    else:
        upper_limit = 100

    if 9e4 < reynolds_sp < 1e6:
        lower_limit = 0.01**x
    elif 1e6 < reynolds_sp:
        lower_limit = 0.1
    else:
        lower_limit = 0.1

    ratio = reynolds_t/reynolds_sp

    if lower_limit < ratio < upper_limit:
        valid = True
    else:
        valid = False

    return {'ratio': ratio, 'valid': valid,
            'lower_limit': lower_limit, 'upper_limit': upper_limit}


def compare_volume_ratio(ratio_sp, ratio_t):
    """Compare volume ratio.

    Compares the volume ratio and evaluates
    them according to the PCT10 criteria.

    Parameters
    ----------
    ratio_sp : float
        Volume ratio from specified condition.
    ratio_t : float
        Volume ratio from test condition.

    Returns
    -------
    Dictionary with ratio (ratio_sp / ratio_t), valid (True if ratio is within limits),
    lower limit and upper limit.
    """
    ratio = ratio_t / ratio_sp

    lower_limit = 0.95
    upper_limit = 1.05

    if lower_limit < ratio < upper_limit:
        valid = True
    else:
        valid = False

    return {'ratio': ratio, 'valid': valid,
            'lower_limit': lower_limit, 'upper_limit': upper_limit}

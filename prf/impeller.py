import numpy as np
from collections import namedtuple
from .curve import *


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
        self.points = points
        self.b = b
        self.D = D
        self.e = e
        self.non_dim_points = []
        for point in points:
            self.non_dim_points.append(NonDimPoint.from_impeller(self, point))

    def flow_coeff(self, flow_m, suc, speed):
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
        v = 1 / suc.rhomass()

        flow_coeff = (flow_m * v /
                      (np.pi**2 * self.D**3 * speed * 15))

        return flow_coeff

    def tip_speed(self, speed):
        """Impeller tip speed.

        Calculates the impeller tip speed for a given speed.

        Parameters
        ----------
        speed : float
            Speed in rad/s.

        Returns
        -------
        tip_speed : float
            Impeller tip speed. (meter**2 radian**2/second**2)

        Examples
        --------

        """
        # TODO check dimensions and units
        return (np.pi * speed * self.D / 60)**2

    def head_coeff(self, head, speed):
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
        return head / self.tip_speed(speed)

    def mach(self, suc, speed):
        return self.D * speed / (2 * suc.speed_sound())

    def reynolds(self, suc, speed):
        return (self.D * self.b * speed * suc.rhomass() /
                (suc.viscosity()))

    @staticmethod
    def volume_ratio(suc, disch):
        return suc.rhomass() / disch.rhomass()

    @convert_to_base_units
    def new_point(self, suc, speed, **kwargs):
        """Curve.

        Calculates a new point based on the given suction state and speed.
        """
        # calculate new head and efficiency
        # use mach to check the best non dim curve to be used
        mach_new = self.mach(suc, speed)

        diff_mach = []
        for point in self.points:
            mach_ = self.mach(point.suc, point.speed)
            diff_mach.append(mach_new - mach_)
        idx = diff_mach.index(min(diff_mach))

        point_old = self.points[idx]
        non_dim_point = self.non_dim_points[idx]

        # store mach, reynolds and volume ratio from original point
        mach_old = self.mach(point_old.suc, point_old.speed)
        reynolds_old = self.reynolds(point_old.suc, point_old.speed)
        volume_ratio_old = self.volume_ratio(point_old.suc, point_old.disch)

        rho = suc.rhomass()
        tip_speed = self.tip_speed(speed)

        # calculate the mass flow
        phi = non_dim_point.flow_coeff
        flow_v = phi * np.pi**2 * self.D**3 * speed / 15
        flow_m = flow_v * rho
        head = non_dim_point.head_coeff * tip_speed
        eff = non_dim_point.eff

        point_new = Point(flow_m=flow_m, speed=speed, suc=suc, head=head, eff=eff)
        # store mach, reynolds and volume ratio from original point
        mach_new = self.mach(point_new.suc, point_new.speed)
        reynolds_new = self.reynolds(point_new.suc, point_new.speed)
        volume_ratio_new = self.volume_ratio(point_new.suc, point_new.disch)

        point_new.mach_comparison = compare_mach(mach_sp=mach_new,
                                                 mach_t=mach_old)
        point_new.reynolds_comparison = compare_reynolds(reynolds_sp=reynolds_new,
                                                         reynolds_t=reynolds_old)

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
    Dictionary with ratio (Ret/Resp), valid (True if ration is within limits),
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


# TODO add compare_reynolds
# TODO add compare_volume_ratio

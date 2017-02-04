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
        curves : list
            List with curve instances.
        b : float
            Impeller width (m).
        D : float
            Impeller diameter (m).
        e : float
            Impeller roughness.
            Defaults to 0.87 um.

        Returns
        -------
        non_dim_curves : list
            List with non dimensional curve instances.

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
        return np.pi * self.D * speed / (60 * suc.speed_sound())

    def points(self, n):
        pass

    @convert_to_base_units
    def new_point(self, suc, speed, **kwargs):
        """Curve.

        Calculates a new curve based on the given suction state and speed.
        """
        # calculate new head and efficiency
        # use mach to check the best non dim curve to be used
        mach_new = self.mach(suc, speed)

        diff_mach = []
        for point in self.points:
            mach_ = self.mach(point.suc, point.speed)
            diff_mach.append(mach_new - mach_)
        idx = diff_mach.index(min(diff_mach))

        non_dim_point = self.non_dim_points[idx]

        rho = suc.rhomass()
        tip_speed = self.tip_speed(speed)

        # calculate the mass flow
        phi = non_dim_point.flow_coeff
        flow_v = phi * np.pi**2 * self.D**3 * speed / 15
        flow_m = flow_v * rho
        head = non_dim_point.head_coeff * tip_speed
        eff = non_dim_point.eff

        return Point(flow_m=flow_m, speed=speed, suc=suc, head=head, eff=eff)

    @classmethod
    def from_non_dimensional_curves(cls, flow, head, efficiency):
        """
        Constructor to initialize an impeller from a non dimensional curve.
        """
        # TODO calculate non dimensional curve
        return cls(flow, head, efficiency)


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


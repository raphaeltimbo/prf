import numpy as np
from collections import namedtuple


class Impeller:
    def __init__(self, curves, b, D, e=0.87e-6):
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
        self.curves = curves
        self.b = b
        self.D = D
        self.e = e
        self.non_dim_curves = []
        for curve in curves:
            self.non_dim_curves.append(NonDimCurve.from_impeller(self, curve))

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

        Parameters:
        -----------
        speed : float
            Speed in rad/s.

        Returns:
        --------
        tip_speed : float
            Impeller tip speed. (meter**2 radian**2/second**2)

        Examples:
        ---------

        """
        return (np.pi * speed * self.D / 60)**2

    def head_coeff(self, head, speed):
        """Head coefficient.

        Calculates the head coefficient given a head and speed.

        Parameters:
        -----------
        head : float
            Head in J/kg.
        speed : float
            Speed in rad/s.

        Returns:
        --------
        head_coeff : float
            Head coefficient (non dimensional).
        """
        return head / self.tip_speed(speed)

    def mach(self, suc, speed):
        return np.pi * self.D * speed / (60 * suc.speed_sound())

    def points(self, n):
        pass

    def new_curve(self, suc, speed):
        """Curve.

        Calculates a new curve based on the given suction state and speed.
        """
        # calculate new head and efficiency
        # use mach to check the best non dim curve to be used
        for curve in self.curves:
            pass

        #flow_v =
        #flow_m = flow_v * suc.rhomass()

        pass

    @classmethod
    def from_non_dimensional_curves(cls, flow, head, efficiency):
        """
        Constructor to initialize an impeller from a non dimensional curve.
        """
        # TODO calculate non dimensional curve
        return cls(flow, head, efficiency)


class NonDimCurve:
    def __init__(self, non_dim_curve):
        # calculate non dimensional curve
        self.non_dim_curve = non_dim_curve
        self.flow_coeff = non_dim_curve[0]
        self.head_coeff = non_dim_curve[1]
        self.efficiency = non_dim_curve[2]

    def points(self):
        point = namedtuple('point', ['flow_coeff', 'head_coeff', 'efficiency'])

        for i in range(len(self.non_dim_curve.T)):
            yield point(flow_coeff=self.flow_coeff[i],
                        head_coeff=self.head_coeff[i],
                        efficiency=self.efficiency[i])

    @classmethod
    def from_impeller(cls, impeller, curve):
        # flow coefficient
        non_dim_curve = np.zeros([3, len(curve.n)])
        # calculate non dim curve and append
        for i, point in enumerate(curve.points()):
            non_dim_curve[0, i] = impeller.flow_coeff(point.flow_m,
                                                      point.suc,
                                                      point.speed)
            non_dim_curve[1, i] = impeller.head_coeff(point.head,
                                                      point.speed)
            non_dim_curve[2, i] = point.efficiency

        return non_dim_curve


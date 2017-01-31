import numpy as np


class Impeller:
    def __init__(self, curves, b, D, e=0.87e-6):
        """
        Impeller instance is initialized with the dimensional curve.

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
        # calculate non dimensional curve
        # flow coefficient
        for curve in curves:
            non_dim_curve = np.zeros([3, len(curve.n)])
            # calculate non dim curve and append
            for point in curve.points():
                non_dim_curve[0, point.n] = self.flow_coeff(point.flow_m,
                                                            point.suc,
                                                            point.speed)
                non_dim_curve[1, point.n] = self.head_coeff(point.head,
                                                            point.speed)
                non_dim_curve[2, point.n] = point.efficiency
            self.non_dim_curves.append(non_dim_curve)

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

    def curve(self, suc):
        pass

    @classmethod
    def from_non_dimensional_curves(cls, flow, head, efficiency):
        """
        Constructor to initialize an impeller from a non dimensional curve.
        """
        # TODO calculate non dimensional curve
        return cls(flow, head, efficiency)

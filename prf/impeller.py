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
            self.non_dim_curves.append(non_dim_curve)

    def flow_coeff(self, flow_m, suc, speed):
        v = 1 / suc.rhomass()

        phi = (flow_m * v /
               (np.pi**2 * self.D**3 * speed * 15))

        return phi

    def tip_speeds(self, speed):
        return (np.pi * speed * self.D / 60)**2



    @classmethod
    def from_non_dimensional_curves(cls, flow, head, efficiency):
        """
        Constructor to initialize an impeller from a non dimensional curve.
        """
        # TODO calculate non dimensional curve
        return cls(flow, head, efficiency)

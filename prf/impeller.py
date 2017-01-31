import numpy as np


class Impeller:
    def __init__(self, curves, b, D, e):
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
        self.non_dim_curves = []
        # calculate non dimensional curve
        # flow coefficient
        for curve in curves:
            # calculate non dim curve and append
            for i in curve:
                pass


    def flow_coeff(self, flow_m, suc, speed):
        v = 1 / suc.rhomass()

        phi = (flow_m * v /
               (np.pi**2 * self.D**3 * speed * 15))


    @classmethod
    def from_non_dimensional_curves(cls, flow, head, efficiency):
        """
        Constructor to initialize an impeller from a non dimensional curve.
        """
        # TODO calculate non dimensional curve
        return cls(flow, head, efficiency)

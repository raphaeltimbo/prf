class Impeller:
    def __init__(self, curves, b, D, e):
        """
        Impeller instance is initialized with the dimensional curves.

        Parameters
        ----------
        curves : list
            List with curves instances.
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

        # calculate non dimensional curves
        # flow coefficient
        self.non_dim_curves = None

    def flow_coeff(self, flow_m, suc):
        pass

    @classmethod
    def from_non_dimensional_curves(cls, flow, head, efficiency):
        """
        Constructor to initialize an impeller from a non dimensional curve.
        """
        # TODO calculate non dimensional curves
        return cls(flow, head, efficiency)

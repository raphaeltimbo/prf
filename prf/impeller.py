class Impeller:
    def __init__(self, curves, b, D):
        """
        Impeller instance is initialized with the dimensional curves.

        Parameters
        ----------
        curves : list
            List with curves instances.
        b : float
            Impeller width.
        D : float
            Impeller diameter.

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
        phi = None

    @classmethod
    def from_non_dimensional_curves(cls, flow, head, efficiency):
        """
        Constructor to initialize an impeller from a non dimensional curve.
        """
        # TODO calculate non dimensional curves
        return cls(flow, head, efficiency)

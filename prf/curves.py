from prf.state import *


class Curves:
    def __init__(self, N, curves, ps, Ts, fluid):
        """
        Construct curves given a speed and an array with
        flow, head and efficiency.
        Parameters
        ----------
        N : float
            Speed for the curves.
        curves : array
            Array with the curves as:
            array([flow],
                  [head],
                  [efficiency])
               ps : float
            Suction pressure.
        Ts : float
            Suction temperature.
        fluid : dict
            Dictionary with constituent and composition
            (e.g.: ({'Oxygen': 0.2096, 'Nitrogen': 0.7812, 'Argon': 0.0092})

        Returns
        -------

        Attributes
        ----------

        Examples
        --------

        """
        self.N = N
        self.curves = curves
        self.ps = ps
        self.Ts = Ts
        self.fluid = fluid

        # construct state
        self.suc_state = State.define('HEOS', self.flow, self.ps, self.Ts)
        self.flow = curves[0]
        self.head = curves[1]
        self.efficiency = curves[2]
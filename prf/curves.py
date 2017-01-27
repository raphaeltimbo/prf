from prf.state import *


class Curves:
    def __init__(self, curves, ps, Ts, fluid):
        """
        Construct curves given a speed and an array with
        flow, head and efficiency.
        Parameters
        ----------
        curves : array
            Array with the curves as:
            array([speed],          -> RPM
                  [flow_m],         -> kg/h
                  [head],           -> J/kg
                  [efficiency])     -> %
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
        self.curves = curves
        self.ps = ps
        self.Ts = Ts
        self.fluid = fluid

        # construct state
        self.suc_state = State.define('HEOS', self.flow_m, self.ps, self.Ts)
        self.speed = curves[0]
        self.flow_m = curves[1]
        self.head = curves[2]
        self.efficiency = curves[3]

    @classmethod
    def from_discharge(cls, curves, ps, Ts, fluid, **kwargs):
        """
                Construct curves given a speed and an array with
        flow, head and efficiency.
        Parameters
        ----------
        curves : array
            Array with the curves as:
            array([speed],          -> RPM
                  [flow_m],         -> kg/h
                  [pd],             -> Pa
                  [Td])             -> K
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
        # calculate head and efficiency for each point



    # TODO add **kwargs for units
    # TODO add constructor -> from_discharge_conditions

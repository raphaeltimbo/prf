import numpy as np
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
        # suction state
        suc = State('HEOS', ps, Ts, fluid, **kwargs)

        # calculate head and efficiency for each point

    # TODO add **kwargs for units
    # TODO add constructor -> from_discharge_conditions


def n_exp(suc, disch):
    """Polytropic exponent.

    Calculates the polytropic exponent given a suction and a discharge state.

    Parameters:
    -----------
    suc : State
        Suction state.
    disch : State
        Discharge state.

    Returns:
    --------
    n_exp : float
        Polytropic exponent.

    Examples:
    ---------

    """
    ps = suc.p()
    vs = 1 / suc.rhomass()
    pd = disch.p()
    vd = 1 / disch.rhomass()

    return np.log(pd/ps)/np.log(vs/vd)


def head_pol(suc, disch):
    """Polytropic head.

    Calculates the polytropic head given a suction and a discharge state.

    Parameters:
    -----------
    suc : State
        Suction state.
    disch : State
        Discharge state.

    Returns:
    --------
    n_exp : float
        Polytropic head.

    Examples:
    ---------

    """
    n = n_exp(suc, disch)

    p2 = disch.p()
    v2 = 1 / disch.rhomass()
    p1 = suc.p()
    v1 = 1 / suc.rhomass()

    return (n/(n-1))*(p2*v2 - p1*v1)

# TODO add tests for n_exp
# TODO add tests for head_pol
# TODO add ef_pol
# TODO add head_isen
# TODO add ef_isen
# TODO add schultz_factor
# TODO add head_pol_schultz

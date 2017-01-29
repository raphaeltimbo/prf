import CoolProp.CoolProp as CP
import pint

ureg = pint.UnitRegistry()
Q_ = pint.UnitRegistry().Quantity


class State(CP.AbstractState):
    # new class to add methods to AbstractState
    def __init__(self, EOS, fluid):
        self.EOS = EOS
        self.fluid = fluid

    @classmethod
    def define(cls, EOS, fluid, p, T, **kwargs):
        """Constructor for state.

        Creates a state and set molar fractions, p and T.

        Parameters
        ----------
        p : float
            The state's pressure
        T : float
            The state's temperature
        fluid : dict
            Dictionary with constituent and composition
            (e.g.: ({'Oxygen': 0.2096, 'Nitrogen': 0.7812, 'Argon': 0.0092})
        EOS : string
            String with HEOS or REFPROP

        Returns
        -------
        state : State object

        Examples:
        ---------
        >>> fluid = {'Oxygen': 0.2096, 'Nitrogen': 0.7812, 'Argon': 0.0092}
        >>> s = State.define('HEOS', fluid, 101008, 273)
        >>> s.rhomass()
        1.2893965217814896
        """
        # get units from kwargs. Set default if not provided.
        p_units = kwargs.get('p_units', ureg.Pa)
        T_units = kwargs.get('T_units', ureg.degK)

        # create unit registers
        p_ = Q_(p, p_units)
        T_ = Q_(T, T_units)

        # convert to base units (SI)
        p_.ito_base_units()
        T_.ito_base_units()

        # define constituents and molar fractions to create and update state
        constituents = []
        molar_fractions = []

        for k, v in fluid.items():
            constituents.append(k)
            molar_fractions.append(v)

        # create an adequate fluid string to cp.AbstractState
        _fluid = '&'.join(constituents)

        state = cls(EOS, _fluid)
        state.set_mole_fractions(molar_fractions)
        state.update(CP.PT_INPUTS, p_.magnitude, T_.magnitude)

        return state


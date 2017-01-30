import CoolProp.CoolProp as CP
import pint
from itertools import combinations

__all__ = ['State', 'fluid_list']

# define pint unit registry
ureg = pint.UnitRegistry()
Q_ = pint.UnitRegistry().Quantity

# apply estimation of binary interaction parameters
mixture = ['CarbonDioxide', 'Nitrogen', 'R134a', 'Oxygen']
for mix1, mix2 in combinations(mixture, 2):
    cas1 = CP.get_fluid_param_string(mix1, 'CAS')
    cas2 = CP.get_fluid_param_string(mix2, 'CAS')
    CP.apply_simple_mixing_rule(cas1, cas2, 'linear')


fluid_list = CP.get_global_param_string('fluids_list').split(',')


def normalize_mix(molar_fractions):
    """
    Normalize the molar fractions so that the sum is 1.

    Parameters:
    -----------
    molar_fractions : list
        Molar fractions of the components.

    Returns:
    --------
    molar_fractions: list
        Molar fractions list will be modified in place.
    """
    total = sum(molar_fractions)
    for i, comp in enumerate(molar_fractions):
        molar_fractions[i] = comp / total

# TODO implement refprop names


class State(CP.AbstractState):
    # new class to add methods to AbstractState
    # no call to super(). see :
    # http://stackoverflow.com/questions/18260095/
    def __init__(self, EOS, fluid):
        self.EOS = EOS
        self.fluid = fluid

    def fluid_dict(self):
        # preserve the dictionary from define method
        fluid_dict = {}
        for k, v in zip(self.fluid_names(), self.get_mole_fractions()):
            fluid_dict[k] = v
        return fluid_dict

    @classmethod
    def define(cls, fluid, p, T, EOS='REFPROP', **kwargs):
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
        >>> s = State.define(fluid, 101008, 273, EOS='HEOS')
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
            if EOS == 'REFPROP':
                k = CP.get_REFPROPname(k)
            constituents.append(k)
            molar_fractions.append(v)

        # create an adequate fluid string to cp.AbstractState
        _fluid = '&'.join(constituents)

        state = cls(EOS, _fluid)
        normalize_mix(molar_fractions)
        state.set_mole_fractions(molar_fractions)
        state.update(CP.PT_INPUTS, p_.magnitude, T_.magnitude)

        return state

    def __copy__(self):
        return self.define(self.fluid_dict(), self.p(), self.T(), self.EOS)

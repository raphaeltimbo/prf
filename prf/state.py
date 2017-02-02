import os
import CoolProp.CoolProp as CP
import pint
from itertools import combinations


__all__ = ['State', 'fluid_list', 'ureg', 'Q_', 'convert_to_base_units', ]

# define pint unit registry
new_units = os.path.join(os.path.dirname(__file__), 'new_units.txt')
ureg = pint.UnitRegistry()
ureg.load_definitions(new_units)
Q_ = ureg.Quantity

# apply estimation of binary interaction parameters
mixture = ['CarbonDioxide', 'Nitrogen', 'R134a', 'Oxygen']
for mix1, mix2 in combinations(mixture, 2):
    cas1 = CP.get_fluid_param_string(mix1, 'CAS')
    cas2 = CP.get_fluid_param_string(mix2, 'CAS')
    CP.apply_simple_mixing_rule(cas1, cas2, 'linear')

CP.set_config_bool(CP.REFPROP_USE_GERG, True)
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


def convert_to_base_units(func):
    """Convert units.

    This function will convert parameters to base units.

    Parameters:
    ----------
    parameters : dict
        Dictionary with parameters and its value.
    units : dict
        Dictionary with the parameter units

    Returns:
    --------
    parameters : dict
        Dictionary with converted units.
    """
    # get units from kwargs. Set default if not provided.
    def inner(*args, units=None, **kwargs):
        if units is None:
            units = {}
        p_units = units.get('p_units', ureg.Pa)
        T_units = units.get('T_units', ureg.degK)
        speed_units = units.get('speed_units', ureg.rad / ureg.s)

        converted_args = []
        for arg_name, value in zip(func.__code__.co_varnames, args):
            if arg_name is 'p':
                p_ = Q_(value, p_units)
                p_.ito_base_units()
                converted_args.append(p_.magnitude)
            if arg_name is 'T':
                T_ = Q_(value, T_units)
                T_.ito_base_units()
                converted_args.append(T_.magnitude)
            if arg_name is 'speed':
                speed_ = Q_(value, speed_units)
                speed_.ito_base_units()
                converted_args.append(speed_.magnitude)
        return func(*converted_args)

    return inner


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
    @convert_to_base_units
    def define(cls, fluid, p, T, EOS='REFPROP', units=None, **kwargs):
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
        #if units is None:
        #    units = {}

        #parameters = {'p': p, 'T': T}
        #converted_values = convert_to_base_units(parameters, units=units)
        #p_ = converted_values['p']
        #T_ = converted_values['T']
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
        state.update(CP.PT_INPUTS, p_, T_)

        return state

    def __copy__(self):
        return self.define(self.fluid_dict(), self.p(), self.T(), self.EOS)

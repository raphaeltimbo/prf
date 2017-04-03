import os
import CoolProp.CoolProp as CP
import pint
from itertools import combinations
from functools import wraps


__all__ = ['State', 'fluid_list', 'ureg', 'Q_', 'convert_to_base_units']

# define pint unit registry
new_units = os.path.join(os.path.dirname(__file__), 'new_units.txt')
ureg = pint.UnitRegistry()
ureg.load_definitions(new_units)
Q_ = ureg.Quantity

CP.set_config_string(CP.ALTERNATIVE_REFPROP_PATH, '/home/raphael/REFPROP-cmake/build/')

mixture = ['CarbonDioxide', 'Nitrogen', 'R134a', 'Oxygen']
for mix1, mix2 in combinations(mixture, 2):
    cas1 = CP.get_fluid_param_string(mix1, 'CAS')
    cas2 = CP.get_fluid_param_string(mix2, 'CAS')
    CP.apply_simple_mixing_rule(cas1, cas2, 'linear')

# list of available fluids
fluid_list = CP.get_global_param_string('fluids_list').split(',')


def normalize_mix(molar_fractions):
    """
    Normalize the molar fractions so that the sum is 1.

    Parameters
    ----------
    molar_fractions : list
        Molar fractions of the components.

    Returns
    -------
    molar_fractions: list
        Molar fractions list will be modified in place.
    """
    total = sum(molar_fractions)
    for i, comp in enumerate(molar_fractions):
        molar_fractions[i] = comp / total


def convert_to_base_units(func):
    """Convert units.

    Decorator used to convert units based on **kwargs.
    """
    # get units from kwargs. Set default if not provided.
    @wraps(func)
    def inner(*args, **kwargs):
        p_units = kwargs.get('p_units', ureg.Pa)
        T_units = kwargs.get('T_units', ureg.degK)
        speed_units = kwargs.get('speed_units', ureg.rad / ureg.s)
        flow_m_units = kwargs.get('flow_m_units', ureg.kg / ureg.s)
        flow_v_units = kwargs.get('flow_v_units', ureg.m**3 / ureg.s)
        power_units = kwargs.get('power_units', ureg.W)

        for arg_name, value in kwargs.items():
            if arg_name == 'p':
                p_ = Q_(value, p_units)
                p_.ito_base_units()
                kwargs[arg_name] = p_.magnitude
            elif arg_name is 'T':
                T_ = Q_(value, T_units)
                T_.ito_base_units()
                kwargs[arg_name] = T_.magnitude
            elif arg_name is 'speed':
                speed_ = Q_(value, speed_units)
                speed_.ito_base_units()
                kwargs[arg_name] = speed_.magnitude
            elif arg_name is 'flow_m':
                flow_m_ = Q_(value, flow_m_units)
                flow_m_.ito_base_units()
                kwargs[arg_name] = flow_m_.magnitude
            elif arg_name is 'flow_v':
                flow_v_ = Q_(value, flow_v_units)
                flow_v_.ito_base_units()
                kwargs[arg_name] = flow_v_.magnitude
            elif arg_name is 'power':
                power_ = Q_(value, power_units)
                power_.ito_base_units()
                kwargs[arg_name] = power_.magnitude

        return func(*args, **kwargs)

    return inner


class State(CP.AbstractState):
    """State class.

    This class is inherited from CP.AbstractState.
    Some extra functionality has been added.
    """
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

    def __repr__(self):
        return 'State: {:.5} Pa @ {:.5} K'.format(self.p(), self.T())

    def __str__(self):
        composition = ''
        for k, v in self.fluid_dict().items():
            composition += '\n {:15}: {:.2f}%'.format(k, v * 100)
        return (
            'State: '
            + composition
            + '\n' + 35*'-'
            + '\n Temperature: {:10.5} K'.format(self.T())
            + '\n Pressure   : {:10.5} Pa'.format(self.p())
            + '\n Density    : {:10.5} kg/m^3'.format(self.rhomass())
            + '\n Enthalpy   : {:10.5} J/kg'.format(self.hmass())
            + '\n Entropy    : {:10.5} J/kg.K'.format(self.smass())
        )

    def __reduce__(self):
        # implemented to enable pickling
        p_ = self.p()
        T_ = self.T()
        fluid_ = self.fluid_dict()
        kwargs = {'p': p_, 'T': T_, 'fluid': fluid_}
        return self._rebuild, (self.__class__, kwargs)

    @staticmethod
    def _rebuild(cls, kwargs):
        p = kwargs.get('p')
        T = kwargs.get('T')
        fluid = kwargs.get('fluid')

        return cls.define(p=p, T=T, fluid=fluid)

    @classmethod
    @convert_to_base_units
    def define(
            cls,
            p=None, T=None, h=None, s=None, d=None,
            fluid=None,
            EOS='REFPROP',
            **kwargs):
        """Constructor for state.

        Creates a state from fluid composition and two properties.
        Properties should be in SI units, **kwargs can be passed
        to change units.

        Parameters
        ----------
        p : float
            State's pressure
        T : float
            State's temperature
        h : float
            State's enthalpy
        s : float
            State's entropy
        d : float

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
        >>> s = State.define(fluid=fluid, p=101008, T=273, EOS='HEOS')
        >>> s.rhomass()
        1.2893965217814896
        """
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

        if p is not None:
            if T is not None:
                state.update(CP.PT_INPUTS, p, T)
            if s is not None:
                state.update(CP.PSmass_INPUTS, p, s)
        if h and s is not None:
            state.update(CP.HmassSmass_INPUTS, h, s)
        if d and s is not None:
            state.update(CP.DmassSmass_INPUTS, d, s)

        return state

    def __copy__(self):
        return self.define(
            p=self.p(),
            T=self.T(),
            fluid=self.fluid_dict(),
            EOS=self.EOS
        )

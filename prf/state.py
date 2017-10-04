import os
import platform
import warnings
import numpy as np
import CoolProp.CoolProp as CP
import CoolProp
import pint
import matplotlib.pyplot as plt
from copy import copy
from itertools import combinations
from functools import wraps
from CoolProp.Plots import PropertyPlot
from CoolProp.Plots.Common import interpolate_values_1d


__all__ = ['State', 'fluid_list', 'ureg', 'Q_', 'convert_to_base_units',
           '__version__CP', '__version__REFPROP']

# define pint unit registry
new_units = os.path.join(os.path.dirname(__file__), 'new_units.txt')
ureg = pint.UnitRegistry()
ureg.load_definitions(new_units)
Q_ = ureg.Quantity

if platform.system() == 'Windows':
    REFPROP_PATH = 'C:/Program Files (x86)/REFPROP'
else:
    REFPROP_PATH = '/home/raphael/REFPROP-cmake/build/'
CP.set_config_string(CP.ALTERNATIVE_REFPROP_PATH, REFPROP_PATH)

mixture = ['CarbonDioxide', 'Nitrogen', 'R134a', 'Oxygen']
for mix1, mix2 in combinations(mixture, 2):
    cas1 = CP.get_fluid_param_string(mix1, 'CAS')
    cas2 = CP.get_fluid_param_string(mix2, 'CAS')
    CP.apply_simple_mixing_rule(cas1, cas2, 'linear')

# list of available fluids
fluid_list = CP.get_global_param_string('fluids_list').split(',')

# versions
__version__CP = CP.get_global_param_string('version')
__version__REFPROP = CP.get_global_param_string('REFPROP_version')


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
        for k, unit in kwargs.items():
            if '_units' in k:
                try:
                    Q_(1, unit)
                except Exception as exc:
                    raise ValueError(f'Wrong units -> {unit}') from exc

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
    To create a State see constructor .define().
    """
    # new class to add methods to AbstractState
    # no call to super(). see :
    # http://stackoverflow.com/questions/18260095/
    def __init__(self, EOS, fluid):
        self.EOS = EOS
        self.fluid = fluid

        # dict relating common properties and their call to CoolProp
        self._prop_dict = dict(Pressure='p', Temperature='T', Enthalpy='hmass',
                               Entropy='smass')

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

        try:
            state = cls(EOS, _fluid)
        except ValueError as exc:
            raise ValueError(
                f'This fluid is not be supported by {EOS}.'
            ) from exc

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

    @classmethod
    @convert_to_base_units
    def defineHA(cls, p=None, T=None, h=None, s=None, d=None,
                 r=None,
                 EOS='REFPROP',
                 **kwargs):
        mol_water = CP.HAPropsSI('Y', 'T', T, 'P', p, 'R', r)

        total = 1 - mol_water

        comp = {
            'Water': mol_water,
            'Argon': total * 0.00935,
            'CO2': total * 0.000319,
            'Nitrogen': total * 0.780840,
            'Oxygen': total * 0.209476,
            }

        return cls.define(p=p, T=T, fluid=comp, EOS=EOS)

    def k(self):
        return self.cpmass() / self.cvmass()

    def kinematic_viscosity(self):
        return self.viscosity() / self.rhomass()

    def _plot_point(self, ax, **kwargs):
        """Plot point.

        Plot point in the given axis. Function will check for axis units and
        plot the point accordingly.

        Parameters
        ----------
        ax : matplotlib.axes, optional
            Matplotlib axes, if None creates a new.

        Returns
        -------
        ax : matplotlib.axes
            Matplotlib axes with plot.
        """
        x_prop = ax.get_xlabel()
        y_prop = ax.get_ylabel()

        for prop in self._prop_dict.keys():
            if prop in x_prop:
                x_value = getattr(self, self._prop_dict[prop])()
            if prop in y_prop:
                y_value = getattr(self, self._prop_dict[prop])()

        # default plot parameters
        kwargs.setdefault('marker', '2')
        kwargs.setdefault('color', 'k')

        ax.scatter(x_value, y_value, **kwargs)

    def plot_envelope(self, ax=None):
        """Plot phase envelop for a given state.

        Parameters
        ----------
        ax : matplotlib.axes, optional
            Matplotlib axes, if None creates a new.

        Returns
        -------
        ax : matplotlib.axes
            Matplotlib axes with plot.
        """
        if ax is None:
            ax = plt.gca()

        # deal with issue #1544
        if self.EOS == 'REFPROP' and len(self.fluid_names()) == 1:
            fluid = self.fluid_dict()
            if 'N2' not in fluid:
                fluid['N2'] = 1e-12
            else:
                fluid['CO2'] = 1e-12

            new_fluid = self.define(p=self.p(), T=self.T(), fluid=fluid)

            # phase envelope
            new_fluid.build_phase_envelope('')
            p_e = new_fluid.get_phase_envelope_data()

        else:
            self.build_phase_envelope('')
            p_e = self.get_phase_envelope_data()

        ax.plot(p_e.T, p_e.p, '-')

        ax.set_xlabel('Temperature $(K)$')
        ax.set_ylabel('Pressure $(Pa)$')
        ax.set_yscale('log')

        return ax

    def plot_ph(self, **kwargs):
        """Plot pressure vs enthalpy."""
        # copy state to avoid changing it
        _self = copy(self)

        # default values for plot
        kwargs.setdefault('unit_system', 'SI')
        kwargs.setdefault('tp_limits', 'ACHP')

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            plot = ModifiedPropertyPlot(_self, 'PH', **kwargs)
            plot.calc_isolines()

        plot.axis.scatter(self.hmass(), self.p(), marker='2',
                          color='k', label=self.__repr__())

        return plot

    def plot_pt(self, **kwargs):
        """Plot pressure vs enthalpy."""
        # copy state to avoid changing it
        _self = copy(self)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            plot = ModifiedPropertyPlot(_self, 'PT', **kwargs)
            plot.calc_isolines()

        plot.axis.scatter(self.T(), self.p(), marker='2',
                          color='k', label=self.__repr__())

        return plot


class ModifiedPropertyPlot(PropertyPlot):
    """Modify CoolProp's property plot."""
    def draw_isolines(self):
        dimx = self._system[self._x_index]
        dimy = self._system[self._y_index]

        sat_props = self.props[CoolProp.iQ].copy()
        if 'lw' in sat_props: sat_props['lw'] *= 2.0
        else: sat_props['lw'] = 1.0
        if 'alpha' in sat_props: min([sat_props['alpha']*2.0,1.0])
        else: sat_props['alpha'] = 1.0

        for i in self.isolines:
            props = self.props[i]
            dew = None; bub = None
            xcrit = None; ycrit = None
            if i == CoolProp.iQ:
                for line in self.isolines[i]:
                    if line.value == 0.0: bub = line
                    elif line.value == 1.0: dew = line
                if dew is not None and bub is not None:
                    xmin, xmax, ymin, ymax = self.get_axis_limits()
                    xmin = dimx.to_SI(xmin)
                    xmax = dimx.to_SI(xmax)
                    ymin = dimy.to_SI(ymin)
                    ymax = dimy.to_SI(ymax)
                    dx = xmax-xmin
                    dy = ymax-ymin
                    dew_filter = np.logical_and(np.isfinite(dew.x),np.isfinite(dew.y))
                    stp = min([dew_filter.size,10])
                    dew_filter[0:-stp] = False
                    bub_filter = np.logical_and(np.isfinite(bub.x),np.isfinite(bub.y))

                    if self._x_index == CoolProp.iP or self._x_index == CoolProp.iDmass:
                        filter_x = lambda x: np.log10(x)
                    else:
                        filter_x = lambda x: x
                    if self._y_index == CoolProp.iP or self._y_index == CoolProp.iDmass:
                        filter_y = lambda y: np.log10(y)
                    else:
                        filter_y = lambda y: y

                    if ((filter_x(dew.x[dew_filter][-1])-filter_x(bub.x[bub_filter][-1])) < 0.050*filter_x(dx) or
                        (filter_y(dew.y[dew_filter][-1])-filter_y(bub.y[bub_filter][-1])) < 0.010*filter_y(dy)):
                        x = np.linspace(bub.x[bub_filter][-1], dew.x[dew_filter][-1], 11)
                        try:
                            y = interpolate_values_1d(
                              np.append(bub.x[bub_filter],dew.x[dew_filter][::-1]),
                              np.append(bub.y[bub_filter],dew.y[dew_filter][::-1]),
                              x_points=x,
                              kind='cubic')
                            self.axis.plot(dimx.from_SI(x),dimy.from_SI(y),**sat_props)
                            warnings.warn("Detected an incomplete phase envelope, fixing it numerically.")
                            xcrit = x[5]
                            ycrit = y[5]
                        except ValueError:
                            continue

            for line in self.isolines[i]:
                if line.i_index == CoolProp.iQ:
                    if line.value == 0.0 or line.value == 1.0:
                        self.axis.plot(dimx.from_SI(line.x),
                                       dimy.from_SI(line.y), **sat_props)
                    else:
                        if xcrit is not None and ycrit is not None:
                            self.axis.plot(
                                dimx.from_SI(np.append(line.x, xcrit)),
                                dimy.from_SI(np.append(line.y, ycrit)),
                                **props)

                else:
                    self.axis.plot(
                        dimx.from_SI(line.x),
                        dimy.from_SI(line.y), **props)

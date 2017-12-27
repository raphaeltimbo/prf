import os
import warnings
import numpy as np
import CoolProp.CoolProp as CP
import CoolProp
import pint
import matplotlib.pyplot as plt
from copy import copy
from pathlib import Path, PosixPath
from itertools import combinations, permutations
from functools import wraps
from CoolProp.Plots import PropertyPlot
from CoolProp.Plots.Common import interpolate_values_1d


__all__ = ['State', 'fluid_list', 'ureg', 'Q_', 'convert_to_base_units',
           '__version__CP', '__version__REFPROP']


############################################################
# Config path and styles
############################################################

# set style and colors
plt.style.use('seaborn-white')
plt.style.use({
    'lines.linewidth': 2.5,
    'axes.grid': True,
    'axes.linewidth': 0.1,
    'grid.color': '.9',
    'grid.linestyle': '--',
    'legend.frameon': True,
    'legend.framealpha': 0.2
})

# define pint unit registry
new_units = os.path.join(os.path.dirname(__file__), 'new_units.txt')
ureg = pint.UnitRegistry()
ureg.load_definitions(new_units)
Q_ = ureg.Quantity


def set_refprop_path(REFPROP_PATH):
    """Sets the refprop path.
    Parameters
    ----------
    REFPROP_PATH : str
        Path to the refprop files.
    """
    CP.set_config_string(CP.ALTERNATIVE_REFPROP_PATH, REFPROP_PATH)


paths = ['C:/Program Files (x86)/REFPROP', '/home/raphael/REFPROP-cmake/build/',
         os.path.join(os.path.dirname(__file__))]

REFPROP_LOADED = False
for path in paths:
    for f in ['/REFPRP64.DLL', '/librefprop.so']:
        file = Path(path + f)
        if file.is_file():
            set_refprop_path(path)
            # check if fluids are there
            path_dirs = set(i.name for i in Path(path).iterdir())
            if isinstance(Path(path), PosixPath):
                if 'fluids' in path_dirs:
                    REFPROP_LOADED = True
                    hmx_file = Path(path + '/fluids/HMX.BNC')
                    break
            else:
                if 'fluids' in path_dirs or 'FLUIDS' in path_dirs:
                    REFPROP_LOADED = True
                    hmx_file = Path(path + '/fluids/HMX.BNC')
                    break

if REFPROP_LOADED is False:
    warnings.warn("Error trying to set REFPROP path.")


mixture = ['CarbonDioxide', 'Nitrogen', 'R134a', 'Oxygen']
for mix1, mix2 in combinations(mixture, 2):
    cas1 = CP.get_fluid_param_string(mix1, 'CAS')
    cas2 = CP.get_fluid_param_string(mix2, 'CAS')
    CP.apply_simple_mixing_rule(cas1, cas2, 'linear')

# list of available fluids
_fluid_list = CP.get_global_param_string('fluids_list').split(',')

# versions
__version__CP = CP.get_global_param_string('version')
__version__REFPROP = CP.get_global_param_string('REFPROP_version')


############################################################
# Fluid names
############################################################


class Fluid:
    def __init__(self, name):
        self.name = name
        self.possible_names = []

    def __repr__(self):
        return f'{type(self).__name__}({self.__dict__}'

# create from _fluid_list
fluid_list = {name: Fluid(name) for name in _fluid_list}

# define possible names
fluid_list['n-Pentane'].possible_names.extend(
    ['pentane', 'n-pentane'])
fluid_list['Isopentane'].possible_names.extend(
    ['isopentane', 'i-pentane'])
fluid_list['n-Hexane'].possible_names.extend(
    ['hexane', 'n-hexane'])
fluid_list['Isohexane'].possible_names.extend(
    ['isohexane', 'i-hexane'])
fluid_list['HydrogenSulfide'].possible_names.extend(
    ['hydrogen sulfide'])
fluid_list['CarbonDioxide'].possible_names.extend(
    ['carbon dioxide'])


def get_name(name):
    """Seach for compatible fluid name."""

    for k, v in fluid_list.items():
        if name in v.possible_names:
            name = k

    fluid_name = CP.get_REFPROPname(name)

    if fluid_name == '':
        raise ValueError(f'Fluid {name} not available. See prf.fluid_list. ')

    return fluid_name


############################################################
# Helper functions
############################################################


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
        head_units = kwargs.get('head_units', ureg.J / ureg.kg)

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
            elif arg_name is 'head':
                head_ = Q_(value, head_units)
                head_.ito_base_units()
                kwargs[arg_name] = head_.magnitude

        return func(*args, **kwargs)

    return inner

############################################################
# State
############################################################


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
                               Entropy='smass', Density='rhomass')

    def fluid_dict(self):
        # preserve the dictionary from define method
        fluid_dict = {}
        for k, v in zip(self.fluid_names(), self.get_mole_fractions()):
            fluid_dict[k] = v
        return fluid_dict

    def not_defined(self):
        """Verifies if the state is defined."""
        if self.T() == -np.infty:
            return True
        else:
            return False

    def update_from_setup_args(self):
        """Update state from setup args."""
        props = {k: v for k, v in self.setup_args.items() if v is not None}
        self.state.update2(**props)

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
        fluid_ = self.fluid_dict()
        kwargs = {k: v for k, v in self.init_args.items() if v is not None}
        kwargs['fluid'] = fluid_
        return self._rebuild, (self.__class__, kwargs)

    @staticmethod
    def _rebuild(cls, kwargs):
        return cls.define(**kwargs)

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

        fluid : dict or str
            Dictionary with constituent and composition
            (e.g.: ({'Oxygen': 0.2096, 'Nitrogen': 0.7812, 'Argon': 0.0092})
            A pure fluid can be created with a string.
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
        >>> # pure fluid
        >>> s = State.define(p=101008, T=273, fluid='CO2')
        >>> s.rhomass()
        1.9716931060214515
        """
        # define constituents and molar fractions to create and update state

        constituents = []
        molar_fractions = []

        # if fluid is a string, consider pure fluid
        if isinstance(fluid, str):
            fluid = {fluid: 1}
        for k, v in fluid.items():
            k = get_name(k)

            constituents.append(k)
            molar_fractions.append(v)

        # create an adequate fluid string to cp.AbstractState
        _fluid = '&'.join(constituents)

        try:
            state = cls(EOS, _fluid)
        except ValueError as exc:
            # check if pair is available at hmx.bnc file
            with open(hmx_file, encoding='latin') as f:
                for pair in permutations(constituents, 2):
                    pair = '/'.join(pair).upper()
                    for line in f:
                        if pair in str(line).upper():
                            continue
                        else:
                            raise ValueError(
                                f'Pair {pair} is not available in HMX.BNC.'
                            ) from exc
            raise ValueError(
                f'This fluid is not be supported by {EOS}.'
            ) from exc

        normalize_mix(molar_fractions)
        state.set_mole_fractions(molar_fractions)
        state.init_args = dict(p=p, T=T, h=h, s=s, d=d)

        if p is not None:
            if T is not None:
                state.update(CP.PT_INPUTS, p, T)
            if s is not None:
                state.update(CP.PSmass_INPUTS, p, s)
            if h is not None:
                state.update(CP.HmassP_INPUTS, h, p)
        if h and s is not None:
            state.update(CP.HmassSmass_INPUTS, h, s)
        if d and s is not None:
            state.update(CP.DmassSmass_INPUTS, d, s)

        return state

    @convert_to_base_units
    def update2(self, **kwargs):
        """Simple state update.

        This method simplifies the state update. Only keyword arguments are
        required to update.

        Parameters
        ----------
        **kwargs : float
            Kwargs with values to update (e.g.: state.update2(p=100200, T=290)
        """
        # TODO add tests to update function.

        inputs = ''.join(k for k in kwargs.keys() if '_units' not in k)

        order_dict = {'Tp': 'pT',
                      'Qp': 'pQ',
                      'sp': 'ps',
                      'ph': 'hp',
                      'Th': 'hT'}

        if inputs in order_dict:
            inputs = order_dict[inputs]

        cp_update_dict = {'pT': CP.PT_INPUTS,
                          'pQ': CP.PQ_INPUTS,
                          'ps': CP.PSmass_INPUTS,
                          'hp': CP.HmassP_INPUTS,
                          'hT': CP.HmassT_INPUTS}

        try:
            cp_update = cp_update_dict[inputs]
        except:
            raise KeyError(f'Update key {inputs} not implemented')

        self.update(cp_update, kwargs[inputs[0]], kwargs[inputs[1]])

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

    def kT(self):
        """Isentropic temperature exponent.

        Calculates the isentropic temperature exponent.

        Returns
        -------
        kT : float
            Isentropic temperature exponent.

        Notes
        -----
        Ludtke pg 52, eq 2.15.
        """
        p = self.p()
        T = self.T()
        dT_dp = self.first_partial_deriv(CP.iT, CP.iP, CP.iSmolar)

        kT = 1 / (1 - (p / T) * (dT_dp))

        return kT

    def kv(self):
        """Isentropic volume exponent.

        Calculates the isentropic volume exponent.

        Returns
        -------
        kT : float
            Isentropic volume exponent.

        Notes
        -----
        Ludtke pg 52, eq 2.9.
        """
        p = self.p()
        rho = self.rhomolar()
        dp_drho = self.first_partial_deriv(CP.iP, CP.iDmolar, CP.iSmolar)

        kv = (rho/p) * (dp_drho)

        return kv

    def kinematic_viscosity(self):
        return self.viscosity() / self.rhomass()

    def z(self):
        """Compressibility factor"""
        z = (self.p() * self.molar_mass()
             / (self.rhomass() * self.gas_constant() * self.T()))

        return z

    def plot_point(self, ax, **kwargs):
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
        kwargs.setdefault('label', self.__repr__())

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

            plot.props[CoolProp.iQ]['lw'] = 0.8
            plot.props[CoolProp.iQ]['color'] = 'k'
            plot.props[CoolProp.iQ]['alpha'] = 0.8

            # isothermal
            plot.props[CoolProp.iT]['lw'] = 0.2
            plot.props[CoolProp.iT]['color'] = 'C0'
            plot.props[CoolProp.iT]['alpha'] = 0.2

            plot.props[CoolProp.iSmass]['lw'] = 0.2
            plot.props[CoolProp.iSmass]['color'] = 'C1'
            plot.props[CoolProp.iSmass]['alpha'] = 0.2

            plot.props[CoolProp.iDmass]['lw'] = 0.2
            plot.props[CoolProp.iDmass]['color'] = 'C2'
            plot.props[CoolProp.iDmass]['alpha'] = 0.2

            plot.calc_isolines()

        self.plot_point(plot.axis)

        return plot

    def plot_pt(self, **kwargs):
        """Plot pressure vs enthalpy."""
        # copy state to avoid changing it
        _self = copy(self)

        # default values for plot
        kwargs.setdefault('unit_system', 'SI')
        kwargs.setdefault('tp_limits', 'ACHP')

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            plot = ModifiedPropertyPlot(_self, 'PT', **kwargs)

            plot.props[CoolProp.iQ]['lw'] = 0.8
            plot.props[CoolProp.iQ]['color'] = 'k'
            plot.props[CoolProp.iQ]['alpha'] = 0.8

            plot.props[CoolProp.iHmass]['lw'] = 0.2
            plot.props[CoolProp.iHmass]['color'] = 'C0'
            plot.props[CoolProp.iHmass]['alpha'] = 0.2

            plot.props[CoolProp.iSmass]['lw'] = 0.2
            plot.props[CoolProp.iSmass]['color'] = 'C1'
            plot.props[CoolProp.iSmass]['alpha'] = 0.2

            plot.props[CoolProp.iDmass]['lw'] = 0.2
            plot.props[CoolProp.iDmass]['color'] = 'C2'
            plot.props[CoolProp.iDmass]['alpha'] = 0.2

            plot.calc_isolines()

        self.plot_point(plot.axis)

        return plot

    def plot_pd(self, **kwargs):
        """Plot pressure vs density."""
        # copy state to avoid changing it
        _self = copy(self)

        # default values for plot
        kwargs.setdefault('unit_system', 'SI')
        kwargs.setdefault('tp_limits', 'ACHP')

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            plot = ModifiedPropertyPlot(_self, 'PD', **kwargs)

            plot.props[CoolProp.iQ]['lw'] = 0.8
            plot.props[CoolProp.iQ]['color'] = 'k'
            plot.props[CoolProp.iQ]['alpha'] = 0.8

            plot.props[CoolProp.iT]['lw'] = 0.2
            plot.props[CoolProp.iT]['color'] = 'C0'
            plot.props[CoolProp.iT]['alpha'] = 0.2

            plot.props[CoolProp.iHmass]['lw'] = 0.2
            plot.props[CoolProp.iHmass]['color'] = 'C1'
            plot.props[CoolProp.iHmass]['alpha'] = 0.2

            plot.props[CoolProp.iSmass]['lw'] = 0.2
            plot.props[CoolProp.iSmass]['color'] = 'C2'
            plot.props[CoolProp.iSmass]['alpha'] = 0.2

            plot.calc_isolines()

        self.plot_point(plot.axis)

        return plot

    def plot_ps(self, **kwargs):
        """Plot pressure vs density."""
        # copy state to avoid changing it
        _self = copy(self)

        # default values for plot
        kwargs.setdefault('unit_system', 'SI')
        kwargs.setdefault('tp_limits', 'ACHP')

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            plot = ModifiedPropertyPlot(_self, 'PS', **kwargs)

            plot.props[CoolProp.iQ]['lw'] = 0.8
            plot.props[CoolProp.iQ]['color'] = 'k'
            plot.props[CoolProp.iQ]['alpha'] = 0.8

            plot.props[CoolProp.iT]['lw'] = 0.2
            plot.props[CoolProp.iT]['color'] = 'C0'
            plot.props[CoolProp.iT]['alpha'] = 0.2

            plot.props[CoolProp.iHmass]['lw'] = 0.2
            plot.props[CoolProp.iHmass]['color'] = 'C1'
            plot.props[CoolProp.iHmass]['alpha'] = 0.2

            plot.props[CoolProp.iDmass]['lw'] = 0.2
            plot.props[CoolProp.iDmass]['color'] = 'C2'
            plot.props[CoolProp.iDmass]['alpha'] = 0.2

            plot.calc_isolines()

        self.plot_point(plot.axis)

        return plot


class ModifiedPropertyPlot(PropertyPlot):
    """Modify CoolProp's property plot."""
    def draw_isolines(self):
        dimx = self._system[self._x_index]
        dimy = self._system[self._y_index]

        sat_props = self.props[CoolProp.iQ].copy()
        if 'lw' in sat_props: sat_props['lw'] *= 2.0
        else: sat_props['lw'] = 1.0
        if 'alpha' in sat_props: min([sat_props['alpha']*1.0,1.0])
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

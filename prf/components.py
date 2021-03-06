import numpy as np
import inspect
from copy import copy, deepcopy
from itertools import chain
from scipy.optimize import newton, root
from .exceptions import OverDefinedSystem, UnderDefinedSystem
from warnings import warn
from .impeller import Impeller
from .point import Point
from .state import State

__all__ = ['Stream', 'Component', 'Mixer', 'Tee', 'Valve',
           'Parameter', 'Compressor', 'ConvergenceBlock']


##################################################
# Helper functions
##################################################

def automatic_docstring(func):
    """Decorator that will automatically generate docstrings."""
    doc = 'Options: \n'

    sig = inspect.signature(func).parameters
    options = sig['options'].default

    for k, v in options.items():
        line = f'{k}: {v} \n'
        doc += line

    def wrapped(option):
        func(option)

    wrapped.__doc__ = doc

    return wrapped


class Parameter:
    """Parameter class.

    This class is used to create objects that hold configuration
    parameters.

    """
    def __init__(self, values):
        """

        Parameters
        ----------
        values : list
            List with options of for the parameter.

        Examples
        --------
        self.pressure_assignment = Parameter(['Equalize All',
                                              'Set Outlet to Lowest Inlet'])
        """
        self.values = values
        try:
            self.current_value = values[0]
        except TypeError:
            self.current_value = values

        kwargs = {k: v for k, v in enumerate(values)}

        @automatic_docstring
        def set_to(options=kwargs):
            self.current_value = self.values[options]

        self.set_to = set_to

    def __repr__(self):
        return str(self.current_value)

##################################################
# Streams
##################################################


class Stream:
    """Material Stream."""
    def __init__(self, name, state=None, flow_m=None):
        """
        A material stream has a flow and a thermodynamic state.

        Parameters
        ----------
        name : str
            Stream name.
        state : prf.State
            Thermodynamic state.
        flow_m : float
            Mass flow

        Examples
        --------
        state0 = prf.State.define(p=1e6, T=300, fluid='CO2')
        stream0 = prf.Stream('s0', state=state0, flow_m=1)
        """
        self.name = name
        self.state = state
        self._flow_m = flow_m

        self.linked_stream = None

    @property
    def flow_m(self):
        if self.linked_stream is None:
            return self._flow_m
        else:
            return self.linked_stream.flow_m

    @flow_m.setter
    def flow_m(self, value):
        """Constrain the mass flow to a value.

        This function should be used during setup of a unit
        to constrain the flow mass value of a stream.
        """
        if isinstance(value, Stream):
            self._flow_m = value.flow_m
            self.linked_stream = value
        else:
            self._flow_m = value

    def break_link(self):
        self.linked_stream = None
        self.flow_m = None

    def __repr__(self):
        return f'\n' \
               f'Stream: {self.name} - \n Flow: {self.flow_m} kg/s - {self.state.__repr__()}'

    def __eq__(self, other):
        eq_flow = (self.flow_m == other.flow_m)
        eq_state = np.array(
            (np.allclose(self.state.p(), other.state.p()),
             np.allclose(self.state.T(), other.state.T()),
             np.allclose(self.state.molar_mass(), other.state.molar_mass())))

        return eq_flow and eq_state.all()

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memo))

        return result

##################################################
# Components
##################################################


class Component:
    """Component (unit)."""
    def __init__(self, name=None):
        """
        Parameters
        ----------
        name : str
            Component name.
        """
        self.name = name
        self.inputs = None
        self.outputs = None
        self.connections = None

        # unknowns
        self.unks = []
        self.x0 = []

    def link(self, inputs, outputs):
        """Connects the components inputs and outputs."""
        self.inputs = inputs
        self.outputs = outputs
        self.connections = list(chain(self.inputs, self.outputs))

    def mass_balance(self):
        """Calculates the mass balance for the component."""
        input_mass = sum((inp.flow_m for inp in self.inputs))
        output_mass = sum((out.flow_m for out in self.outputs))

        return input_mass - output_mass

    def energy_balance(self):
        """Calculates the energy balance for the component."""
        input_energy = 0
        for inp in self.inputs:
            inp_energy = inp.flow_m * inp.state.hmass()
            input_energy += inp_energy

        output_energy = 0
        for out in self.outputs:
            out_energy = out.flow_m * out.state.hmass()
            output_energy += out_energy

        input_mass = sum((inp.flow_m for inp in self.inputs))
        output_mass = sum((out.flow_m for out in self.outputs))

        return (input_energy / input_mass) - (output_energy / output_mass)

    def setup(self):
        """Setup constraints."""

    def balance(self, x):
        """Update each stream with iteration value."""
        for i, unk in enumerate(self.unks):
            s_name, prop = unk.split('_', maxsplit=1)
            for con in self.connections:
                if con.name == s_name:
                    if prop == 'flow_m':
                        con.flow_m = x[i]
                    else:
                        con.state.setup_args[prop] = x[i]
                props = {k: v for k, v in con.state.setup_args.items() if v is not None}
                if len(props) == 2:
                    try:
                        con.state.update2(**props)
                    except ValueError:
                        # if refprop does not converge, try CP's HEOS
                        heos_state = State.define(**props, fluid=con.state.fluid_dict(), EOS='HEOS')
                        heos_state.setup_args = con.state.setup_args
                        con.state = heos_state

        y = np.zeros_like(x)

        if len(self.unks) == 1:
            if 'flow_m' in self.unks[0]:
                y[0] = self.mass_balance()
            else:
                y[0] = self.energy_balance()

        else:
            y[0] = self.mass_balance()
            y[1] = self.energy_balance()

        return y

    def check_consistency(self):
        """Check component consistency"""
        if len(self.unks) < 2:
            raise OverDefinedSystem(f'Unit {self.name} is over defined. Unknowns : {self.unks}')
        elif len(self.unks) > 2:
            raise UnderDefinedSystem(f'Unit {self.name} is under defined for {self.name}.'
                                     f' Unknowns : {self.unks}')

    def set_x0(self):
        """Set initial values for convergence."""
        x0 = self.x0
        for unk in self.unks:
            if unk[-1] == 'p':
                x0.append(100000.)
            if unk[-1] == 'T':
                x0.append(300.)
            if unk[-1] == 'm':
                x0.append(1.)

    def run(self):
        """Run the unit and calculate unknowns."""
        # apply constraints
        self.setup()
        # check all unknowns
        for con in self.connections:
            if con.flow_m is None \
                    and con.linked_stream is None\
                    and (con.name + '_flow_m') not in self.unks:
                self.unks.append(con.name + '_flow_m')

            if con.state.not_defined:
                if con.state.setup_args['p'] is None\
                        and (con.name + '_p') not in self.unks:
                    self.unks.append(con.name + '_p')
                if con.state.setup_args['T'] is None\
                        and (con.name + '_T') not in self.unks:
                    self.unks.append(con.name + '_T')

        self.check_consistency()

        self.set_x0()

        if len(self.unks) == 0:
            if np.allclose(self.balance([0, 0]), np.array([0, 0])) is False:
                raise OverDefinedSystem(f'System is over defined and'
                                        f'not balanced {self.balance(0)}')
        else:
            root(self.balance, self.x0)

    def is_solved(self):
        for con in self.connections:
            if con.flow_m is None:
                return False
            elif con.state.not_defined():
                return False
        return True

    def __repr__(self):
        return f'Inputs: \n {self.inputs} \n' \
               f'Outputs: \n {self.outputs} \n'


class Mixer(Component):
    def __init__(self, name):
        """A mixer.

        A mixer has inputs and outputs streams.
        Undetermined stream values are solved through balance
        of mass and energy.

        """
        self.pressure_assignment = Parameter(['Equalize All',
                                              'Set Outlet to Lowest Inlet'])
        super().__init__(name)

    def setup(self):
        pressure = []
        for con in self.connections:
            if con.state.init_args['p'] is not None:
                pressure.append(con.state.init_args['p'])

        if self.pressure_assignment.current_value == 'Equalize All':
            for con in self.connections:
                if con.state.init_args['p'] is None:
                    con.state.setup_args['p'] = pressure[0]
                if len(pressure) > 1:
                    raise OverDefinedSystem('System is over determined for '
                                            '"Equalize All"')

        if self.pressure_assignment.current_value == 'Set Outlet to Lowest Inlet':
            for con in self.connections:
                if con.state.init_args['p'] is None:
                    con.state.setup_args['p'] = min(pressure)


class Tee(Component):
    """Tee operation.

    Splits feed stream into multiple outputs with the same conditions and
    composition.
    """
    def setup(self):
        pressure = {}
        temperature = {}

        for con in self.connections:
            if con.state.init_args['p'] is not None:
                pressure[f'{con.name}_p'] = con.state.init_args['p']
            if con.state.init_args['T'] is not None:
                temperature[f'{con.name}_T'] = con.state.init_args['T']

        if len({i for i in pressure.values()}) > 1:
            raise OverDefinedSystem(f'System is over defined {pressure}')
        if len({i for i in temperature.values()}) > 1:
            raise OverDefinedSystem(f'System is over defined {temperature}')

        for con in self.connections:
            con.state.setup_args['p'] = [i for i in pressure.values()][0]
            con.state.setup_args['T'] = [i for i in temperature.values()][0]

    def check_consistency(self):
        pass


class Valve(Component):
    """Simple valve.

    Valve that will give an isenthalpic expansion.
    """
    def __init__(self, name, cv=None, v_open=0.5):
        """

        Parameters
        ----------
        name : str
            Valve name
        cv : float
            Valve cv.
        v_open : float
            Valve opening.
        """
        self.cv = cv
        self.v_open = v_open

        self.init_cv = deepcopy(cv)

        super().__init__(name)

    def setup(self):
        # same input and output mass
        inp = self.inputs[0]
        out = self.outputs[0]

        # constrain mass
        if out.flow_m is None:
            out.flow_m = inp
        elif inp.flow_m is None:
            inp.flow_m = out
        else:
            if out.flow_m != inp.flow_m:
                raise OverDefinedSystem(f'Different mass for {inp} and {out}')

    def calc_cv(self):
        """Calculate cv based on simple resistance equation."""
        m = self.inputs[0].flow_m
        v_open = self.v_open
        dP = self.inputs[0].state.p() - self.outputs[0].state.p()
        rho = self.inputs[0].state.rhomass()
        cv = self.cv

        return cv - m / np.sqrt(v_open * dP * rho)

    def balance(self, x):
        for i, unk in enumerate(self.unks):
            s_name, prop = unk.split('_', maxsplit=1)
            if s_name == 'valve':
                self.cv = x[i]
            for con in self.connections:
                if con.name == s_name:
                    if prop == 'flow_m':
                        con.flow_m = x[i]
                    else:
                        con.state.setup_args[prop] = x[i]

                props = {k: v for k, v in con.state.setup_args.items() if v is not None}
                if len(props) == 2:
                    try:
                        con.state.update2(**props)
                    except ValueError:
                        # if refprop does not converge, try CP's HEOS
                        heos_state = State.define(**props, fluid=con.state.fluid_dict(), EOS='HEOS')
                        heos_state.setup_args = con.state.setup_args
                        con.state = heos_state

        y = np.zeros_like(x, dtype=np.float)

        # mass balance is already satisfied for a valve

        y[0] = self.energy_balance()
        y[1] = self.calc_cv()

        return y

    def run(self):
        if self.cv is None:
            self.unks.append('valve_cv')
            self.x0.append(0.8)

        super().run()


class Compressor(Component):
    """Compressor."""
    def __init__(self, name, impeller=None, speed=None, flow_m=None, b=None, D=None):
        """
        # TODO change this to impeller or and create compressor class
        Parameters
        ----------
        name : str
            Compressor name.
        impeller : prf.Impeller
            Impeller object.
        speed : float
            Impeller speed.
        flow_m : float
            Mass flow.
        b : float
            Impeller width
        D : float
            Impeller diameter.
        """
        self.name = name
        self.init_impeller = impeller
        self.impeller = None
        self.speed = speed
        self.b = b
        self.D = D
        super().__init__(name)

    def setup(self):
        # same input and output mass
        inp = self.inputs[0]
        out = self.outputs[0]

        # constrain mass
        if out.flow_m is None:
            out.flow_m = inp
        elif inp.flow_m is None:
            inp.flow_m = out
        else:
            if out.flow_m != inp.flow_m:
                raise OverDefinedSystem(f'Different mass for {inp} and {out}')

    def check_consistency(self):
        pass

    def run(self):
        inp = self.inputs[0]
        out = self.outputs[0]

        if self.init_impeller is None:
            point = Point(speed=self.speed, flow_m=inp.flow_m, suc=inp.state, disch=out.state)
            self.impeller = Impeller(point, b=self.b, D=self.D)
        else:
            self.impeller.suc = inp.state
            out.state.update2(p=self.impeller.disch.p(), T=self.impeller.disch.T())


class ConvergenceBlock(Component):
    """Convergence block."""
    def __init__(self, stream, units):
        """
        A convergence block is used to converge the recycles.
        The selected stream is split in two streams (sc0 and sc1).
        A guess is made for one of the streams (sc0). Units are calculated
        and the values for the other stream (sc1) are compared with the
        guessed stream (sc0).

        Parameters
        ----------
        stream : str
            Name of the stream that will be used for iteration.
        units : list
            List with prf.Component objects
        """
        self.stream = stream
        self.units = units
        self.units0 = deepcopy(units)

        self.converged_units = None

        # convergence information
        self.tolerance = 0.1
        self.iter = 0
        self.convergence_info = None
        self.converged = False
        self.y0 = None
        self.y1 = None

        super().__init__('ConvBlock')

    def setup(self):
        # create convergence block
        sc = 0
        for unit in self.units0:
            new_inputs = []
            for i, inp in enumerate(unit.inputs):
                if inp.name == self.stream:
                    new_inp = deepcopy(inp)
                    new_inp.name = f'sc{sc}'
                    sc += 1
                    new_inputs.append(new_inp)
                else:
                    new_inputs.append(inp)

            new_outputs = []
            for i, out in enumerate(unit.outputs):
                if out.name == self.stream:
                    new_out = deepcopy(out)
                    new_out.name = f'sc{sc}'
                    sc += 1
                    new_outputs.append(new_out)
                else:
                    new_outputs.append(out)

            unit.link(inputs=new_inputs, outputs=new_outputs)

        for unit in self.units0:
            for con in unit.connections:
                if 'sc' in con.name:
                    setattr(self, con.name, con)

    def balance(self, x):
        # store units that are not solved and go forward
        not_solved_units = []

        for unit in self.units0:
            for con in unit.connections:
                if con.name == 'sc0':
                    con.flow_m = x[0]
                    con.state.setup_args['T'] = x[1]

                    props = {k: v for k, v in
                             con.state.setup_args.items() if v is not None}
                    if len(props) == 2:
                        try:
                            con.state.update2(**props)
                        except ValueError:
                            # if refprop does not converge, try CP's HEOS
                            heos_state = State.define(
                                **props, fluid=con.state.fluid_dict(), EOS='HEOS')
                            heos_state.setup_args = con.state.setup_args
                            con.state = heos_state

                if con.name == 'sc1':
                    con.state.setup_args['p'] = self.sc0.state.setup_args['p']

                    props = {k: v for k, v in
                             con.state.setup_args.items() if v is not None}
                    if len(props) == 2:
                        try:
                            con.state.update2(**props)
                        except ValueError:
                            # if refprop does not converge, try CP's HEOS
                            heos_state = State.define(
                                **props, fluid=con.state.fluid_dict(), EOS='HEOS')
                            heos_state.setup_args = con.state.setup_args
                            con.state = heos_state

            # if cv is given, calculate mass
            if isinstance(unit, Valve):
                if unit.init_cv is not None:
                    # break connections
                    for con in unit.connections:
                        con.break_link()

            unit.run()
            if not unit.is_solved():
                not_solved_units.append(unit)

        # try to go back and run unsolved units
        for unit in not_solved_units:
            unit.run()

        y = np.zeros_like(x)

        for unit in self.units0:
            for con in unit.connections:
                if con.name == 'sc1':
                    y[0] = con.flow_m - x[0]
                    y[1] = con.state.T() - x[1]

        return y

    def run(self):
        self.setup()

        for unit in self.units0:
            unit.setup()
            for con in unit.connections:
                try:
                    con.state.update_from_setup_args()
                except KeyError:
                    continue

        self.convergence_info = root(self.balance, [0.8, 300])

        for unit in self.units0:
            for con in unit.connections:
                if 'sc' in con.name:
                    con.name = self.stream

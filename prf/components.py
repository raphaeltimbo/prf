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

__all__ = ['Stream', 'Component', 'Mixer', 'Valve', 'Parameter', 'Compressor',
           'ConvergenceBlock']


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
    def __init__(self, name, state=None, flow_m=None):
        self.name = name
        self.state = state
        self._flow_m = flow_m

        # setup args will initially be set to init_args.
        # later this attribute can be used to store args
        # defined during the setup process.
        self.state.setup_args = copy(state.init_args)

        # self.constrained_mass = False
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
        result.state.setup_args = self.state.setup_args

        return result

##################################################
# Components
##################################################


class Component:
    def __init__(self, name=None):
        self.name = name
        self.inputs = None
        self.outputs = None
        self.connections = None

        # unknowns
        self.unks = []
        self.x0 = []

    def link(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs
        self.connections = list(chain(self.inputs, self.outputs))

    def mass_balance(self):
        input_mass = sum((inp.flow_m for inp in self.inputs))
        output_mass = sum((out.flow_m for out in self.outputs))

        return input_mass - output_mass

    def energy_balance(self):
        input_energy = 0
        for inp in self.inputs:
            inp_energy = inp.flow_m * inp.state.hmass()
            input_energy += inp_energy

        output_energy = 0
        for out in self.outputs:
            out_energy = out.flow_m * out.state.hmass()
            output_energy += out_energy

        total_mass = sum((inp.flow_m for inp in self.inputs))

        return (input_energy / total_mass) - (output_energy / total_mass)

    def setup(self):
        """setup constraints"""

    def balance(self, x):
        """update each stream with iteration value"""
        for i, unk in enumerate(self.unks):
            s_name, prop = unk.split('_', maxsplit=1)
            for con in self.connections:
                if con.name == s_name:
                    if prop == 'flow_m':
                        con.flow_m = x[i]
                    else:
                        con.state.setup_args[prop] = x[i]
                props = {k: v for k, v in con.state.setup_args.items() if v is not None}
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
        """Check system consistency"""
        if len(self.unks) < 2:
            raise OverDefinedSystem(f'System is over defined. Unknowns : {self.unks}')
        elif len(self.unks) > 2:
            raise UnderDefinedSystem(f'System is under defined. Unknowns : {self.unks}')

    def set_x0(self):
        x0 = self.x0
        for unk in self.unks:
            if unk[-1] == 'p':
                x0.append(100000)
            if unk[-1] == 'T':
                x0.append(300)
            if unk[-1] == 'm':
                x0.append(1)

    def run(self):
        # apply constraints
        self.setup()
        # check all unknowns
        for con in self.connections:
            if con.flow_m is None and con.linked_stream is None:
                self.unks.append(con.name + '_flow_m')

            if con.state.not_defined:
                if con.state.setup_args['p'] is None:
                    self.unks.append(con.name + '_p')
                if con.state.setup_args['T'] is None:
                    self.unks.append(con.name + '_T')

        self.check_consistency()

        self.set_x0()

        root(self.balance, self.x0)

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


class Valve(Component):
    """Simple valve.

    Valve that will give an isenthalpic expansion.
    """
    def __init__(self, name, cv=None, v_open=0.5):
        self.cv = cv
        self.v_open = v_open

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
                    try:
                        con.state.update2(**props)
                    except ValueError:
                        # if refprop does not converge, try CP's HEOS
                        heos_state = State.define(**props, fluid=con.state.fluid_dict(), EOS='HEOS')
                        heos_state.setup_args = con.state.setup_args
                        con.state = heos_state

        y = np.zeros_like(x)

        # mass balance is already satisfied for a valve
        y[0] = self.energy_balance()
        y[1] = self.calc_cv()

        return y

    def run(self):
        if self.cv is None:
            self.unks.append('valve_cv')
            self.x0.append(0.1)

        super().run()


class Compressor(Component):
    def __init__(self, impeller=None, speed=None, flow_m=None, b=None, D=None):
        self.init_impeller = impeller
        self.impeller = None
        self.speed = speed
        self.b = b
        self.D = D
        super().__init__()

    def run(self):
        inp = self.inputs[0]
        out = self.outputs[0]

        if self.init_impeller is None:
            point = Point(speed=self.speed, flow_m=inp.flow_m, suc=inp.state, disch=out.state)
            self.impeller = Impeller(point, b=self.b, D=self.D)
        else:
            self.impeller.suc = inp.state
            out.state.update2(p=self.impeller.disch.p(), T=self.impeller.disch.T())

        super().run()


class ConvergenceBlock(Component):
    def __init__(self, stream, units):
        self.stream = stream
        self.units = units

        self._units = None

        self.converged_units = None

        # convergence information
        self.tolerance = 0.1
        self.iter = 0
        self.converged = False
        self.y0 = None
        self.y1 = None
        super().__init__()

    def converge(self, new_x):
        # initialize
        if self.iter == 0:
            self._units = deepcopy(self.units)

        units0 = deepcopy(self._units)

        # select stream
        for unit in units0:
            for stream in unit.connections:
                if stream.name == self.stream:
                    s0 = deepcopy(stream)
                    s1 = deepcopy(stream)

        # select prop
        s0.state.setup_args['T'] = new_x

        for unit in units0:
            new_inputs = []
            for i, inp in enumerate(unit.inputs):
                if inp.name == self.stream:
                    new_inputs.append(s0)
                else:
                    new_inputs.append(inp)

            new_outputs = []
            for i, out in enumerate(unit.outputs):
                if out.name == self.stream:
                    new_outputs.append(s1)
                else:
                    new_outputs.append(out)

            unit.link(inputs=new_inputs, outputs=new_outputs)

        for unit in units0:
            unit.run()

        self.y0 = s0.state.T()
        self.y1 = s1.state.T()

        self.iter += 1

        self.converged_units = units0

        return self.y1 - self.y0

    def run(self):
        newton(self.converge, 300)


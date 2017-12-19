import numpy as np
import inspect
from copy import copy, deepcopy
from itertools import chain
from scipy.optimize import newton
from .exceptions import MassError, OverDefinedWarning
from warnings import warn
from .impeller import Impeller
from .point import Point

__all__ = ['Stream', 'Component', 'Mixer', 'Valve', 'Parameter']


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
    def __init__(self, name=None, state=None, flow_m=None):
        self.name = name
        self.state = state
        self.flow_m = flow_m

        # setup args will initially be set to init_args.
        # later this attribute can be used to store args
        # defined during the setup process.
        self.state.setup_args = copy(state.init_args)

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
    def __init__(self):
        self.inputs = None
        self.outputs = None
        self.connections = None

        # unknown mass and state
        self.unk_mass = None
        self.unk_state = None
        self.prop = None
        self.var_prop = None
        self.energy_x0 = None

        self.total_mass = None

    def link(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs
        self.connections = list(chain(self.inputs, self.outputs))
        self.setup()

    def setup(self):
        pass

    def get_unk_mass(self):
        unk_mass = []

        for link in chain(self.inputs, self.outputs):
            if link.flow_m is None:
                unk_mass.append(link)
        if len(unk_mass) > 1:
            raise MassError(f'More than one undetermined mass {unk_mass}')
        elif len(unk_mass) == 0:
            # check if mass balance is satisfied
            input_mass = sum((inp.flow_m for inp in self.inputs))
            output_mass = sum((out.flow_m for out in self.outputs))
            if np.allclose(input_mass, output_mass):
                self.total_mass = input_mass
            else:
                raise MassError(f'Error in mass balance input:{input_mass} kg/s'
                                f' output:{output_mass} kg/s')
        else:
            self.unk_mass = unk_mass[0]

        # solve first for mass
    def mass_balance(self, new_mass):
        unk_mass = self.unk_mass
        unk_mass.flow_m = new_mass
        input_mass = sum((inp.flow_m for inp in self.inputs))
        output_mass = sum((out.flow_m for out in self.outputs))

        return input_mass - output_mass

    def get_unk_state(self):
        # search for undetermined state
        unk_state = []

        for link in chain(self.inputs, self.outputs):
            if link.state.p() == -np.infty or link.state.p() == np.infty:
                unk_state.append(link)
            else:
                #  record a state as reference for initial guess
                ref_state_args = link.state.setup_args

        if len(unk_state) > 1:
            raise ValueError(f'More than one undetermined state {unk_state}')

        try:
            unk_state = unk_state[0].state
        except IndexError:
            # unk_state is None if list is empty
            unk_state = None

        self.unk_state = unk_state

        if self.unk_state is not None:
            self.prop = {k: v for k, v in
                         unk_state.setup_args.items() if v is not None}
            self.var_prop = 'T' if 'p' in self.prop else 'p'
            # initial guess based on value of some of the links
            self.energy_x0 = ref_state_args[self.var_prop]

    def energy_balance(self, new_value, var_prop):
        # get available property
        unk_state = self.unk_state
        # var_prop -> property that will change to obtain convergence

        prop = self.prop
        prop[var_prop] = new_value

        unk_state.update2(**prop)

        input_energy = 0
        for inp in self.inputs:
            inp_energy = inp.flow_m * inp.state.hmass()
            input_energy += inp_energy

        output_energy = 0
        for out in self.outputs:
            out_energy = out.flow_m * out.state.hmass()
            output_energy += out_energy

        return (input_energy / self.total_mass) - (output_energy / self.total_mass)

    def run(self):

        self.get_unk_mass()
        if self.total_mass is None:
            newton(self.mass_balance, 0)
            self.total_mass = sum((inp.flow_m for inp in self.inputs))

        # solve for energy
        self.get_unk_state()
        if self.unk_state is not None:
            newton(self.energy_balance, self.energy_x0, args=(self.var_prop,))

        for i, inp in enumerate(self.inputs):
            setattr(self, f'inp{i}', inp)
        for i, out in enumerate(self.outputs):
            setattr(self, f'out{i}', out)


class Mixer(Component):
    def __init__(self):
        """A mixer.

        A mixer has inputs and outputs streams.
        Undetermined stream values are solved through balance
        of mass and energy.

        """
        # total mass will be calculated after call to run()
        self.pressure_assignment = Parameter(['Equalize All',
                                              'Set Outlet to Lowest Inlet'])

        super().__init__()

    def setup(self):
        pressure = []
        temperature = []
        for con in self.connections:
            for k, v in con.state.init_args.items():
                if k == 'p' and v is not None:
                    pressure.append(v)
                if k == 'T' and v is not None:
                    temperature.append(v)

        if self.pressure_assignment.current_value == 'Equalize All':
            if len(pressure) > 1:
                warn(f'Pressure of streams are over defined'
                     f' for {self.pressure_assignment}',
                     OverDefinedWarning)

            for con in self.connections:
                con.state.setup_args['p'] = pressure[0]
                if len(self.inputs) == 1:
                    con.state.setup_args['T'] = temperature[0]
                if (con.state.not_defined and
                        con.state.init_args != con.state.setup_args):
                    props = {k: v for k, v in con.state.setup_args.items() if v is not None}
                    if len(props) == 2:
                        con.state.update2(**props)

        elif (self.pressure_assignment.current_value
              == 'Set Outlet to Lowest Inlet'):
            out_state = self.outputs[0].state
            if out_state.init_args['p'] is not None:
                warn(f'Pressure of streams are over defined'
                     f' for {self.pressure_assignment}',
                     OverDefinedWarning)
            out_state.setup_args['p'] = min(pressure)
            props = {k: v for k, v in out_state.setup_args.items() if v is not None}
            if len(props) == 2:
                out_state.update2(**props)


class Valve(Component):
    """Simple valve.

    Valve that will give an isenthalpic expansion.
    """

    def __init__(self, cv=None):
        self.cv = cv
        super().__init__()

    def p_d(self):
        inp = self.inputs[0]
        m = inp.flow_m
        p_u = inp.state.p()
        T_u = inp.state.T()
        MW = inp.state.molar_mass()
        z_u = inp.state.z()
        cv = self.cv

        p_d = p_u * np.sqrt(1 - (z_u * T_u / MW) * (m / (cv * p_u)) ** 2)

        return p_d

    def calc_mass_flow(self):
        inp = self.inputs[0]
        out = self.outputs[0]

        p_u = inp.state.p()
        p_d = out.state.init_args['p']
        T_u = inp.state.T()
        MW = inp.state.molar_mass()
        z_u = inp.state.z()
        cv = self.cv

        m = (cv * p_u * np.sqrt(1 - (p_d / p_u) ** 2)
             / (np.sqrt(z_u * T_u / MW)))

        return m

    def run(self):
        inp = self.inputs[0]

        try:
            self.get_unk_mass()
        except MassError:
            inp.flow_m = self.calc_mass_flow()

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

        # convergence information
        self.tolerance = 0.1
        self.iter = 0
        self.converged = False
        self.y0 = None
        self.y1 = None
        super().__init__()

    def run(self, new_x):
        print(self.iter)
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

        return self.y1 - self.y0


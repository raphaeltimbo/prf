import numpy as np
from itertools import chain
from scipy.optimize import newton
from .exceptions import MassError

__all__ = ['Stream', 'Component', 'Mixer', 'Valve']


class Stream:
    def __init__(self, state=None, flow_m=None):
        self.state = state
        self.flow_m = flow_m

    def __repr__(self):
        return f'Flow: {self.flow_m} kg/s - {self.state.__repr__()}'


class Component:
    def __init__(self):
        self.inputs = None
        self.outputs = None

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
                ref_state_args = link.state.init_args

        if len(unk_state) > 1:
            raise ValueError(f'More than one undetermined state {unk_state}')

        unk_state = unk_state[0].state
        self.prop = {k: v for k, v in
                     unk_state.init_args.items() if v is not None}
        self.var_prop = 'T' if 'p' in self.prop else 'p'
        # initial guess based on value of some of the links
        self.energy_x0 = ref_state_args[self.var_prop]
        self.unk_state = unk_state

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
        super().__init__()


class Valve(Component):
    """Simple valve.

    Valve that will give an isenthalpic expansion.
    """

    def __init__(self, cv):
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


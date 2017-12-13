import numpy as np
from itertools import chain
from scipy.optimize import newton


class Stream:
    def __init__(self, state=None, flow_m=None):
        self.state = state
        self.flow_m = flow_m


class Mixer:
    def __init__(self):
        """A mixer.

        A mixer has inputs and outputs streams.
        Undetermined stream values are solved through balance
        of mass and energy.

        """
        self.inputs = None
        self.outputs = None

        # total mass will be calculated after call to run()
        self.total_mass = None

    def run(self):
        # calculate mass balance
        # search for undetermined mass
        unk_mass = []
        for link in chain(self.inputs, self.outputs):
            if link.flow_m is None:
                unk_mass.append(link)
        if len(unk_mass) > 1:
            raise ValueError(f'More than one undetermined mass {unk_mass}')

        unk_mass = unk_mass[0]

        # solve first for mass
        def mass(new_mass):
            unk_mass.flow_m = new_mass
            input_mass = sum((inp.flow_m for inp in self.inputs))
            output_mass = sum((out.flow_m for out in self.outputs))

            return input_mass - output_mass

        newton(mass, 0)

        self.total_mass = sum((inp.flow_m for inp in self.inputs))

        # search for undetermined state
        unk_state = []
        for link in chain(self.inputs, self.outputs):
            if link.state.p() == -np.infty or link.state.p() == np.infty:
                unk_state.append(link)
        if len(unk_state) > 1:
            raise ValueError(f'More than one undetermined mass {unk_mass}')
        unk_state = unk_state[0]

        # solve for energy
        def energy(new_T):
            # get available property
            prop = {k: v for k, v in unk_state.init_args.items() if v is not None}
            unk_state.state.update2(T=new_T, **prop)

            input_energy = 0
            for inp in self.inputs:
                energy = inp.flow_m * inp.state.hmass()
                input_energy += energy
            output_energy = 0
            for out in self.outputs:
                energy = out.flow_m * out.state.hmass()
                output_energy += energy

            return (input_energy / self.total_mass) - (output_energy / self.total_mass)

        newton(energy, 300)

        for i, inp in enumerate(self.inputs):
            setattr(self, f'inp{i}', inp)
        for i, out in enumerate(self.outputs):
            setattr(self, f'out{i}', out)

    def link(self, inputs=None, outputs=None):
        """Define inputs and outputs"""
        self.inputs = inputs
        self.outputs = outputs

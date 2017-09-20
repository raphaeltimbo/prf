"""
prf
===

Package to evaluate the performance of a centrifugal compressor.
Example of use:

Define the fluid as a dictionary:

fluid = {'CarbonDioxide': 0.79585,
         'R134a': 0.16751,
         'Nitrogen': 0.02903,
         'Oxygen': 0.00761}

Input units as a dictionary:

units = {'p_units': 'bar', 'T_units': 'degK', 'speed_units': 'RPM',
         'flow_m_units': 'kg/h'}

Define suction and discharge states:

suc = State.define(fluid=fluid, p=2.238, T=298.3, **units)
disch = State.define(fluid=fluid, p=7.255, T=391.1, **units)

Create performance point(s):

point = Point(suc=suc, disch=disch, speed=7941, flow_m=34203.6, **units)

Create an impeller that will hold and convert curves.

imp = Impeller(point, b=0.0285, D=0.365)
"""
from .impeller import *
from .state import *
from .point import *
from .results import *

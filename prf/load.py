import yaml
from prf.state import State
from prf.point import Point, Curve
from prf.impeller import Impeller


__all__ = ['load']


def load(file):
    """Load impeller from yaml file.

    Spreadsheet or other file types should be converted to yaml files before
    loading.


    """
    with open(file, 'r') as f:
        input_data = yaml.load(f)

    units = {}

    for k, v in input_data['units'].items():
        units[k + '_units'] = v

    if units['T_units'] == 'C':
        units['T_units'] = 'degC'

    flow_v = input_data['curve']['flow']
    head = input_data['curve']['head']
    eff = input_data['curve']['eff']
    ps = input_data['suction']['ps']
    Ts = input_data['suction']['Ts']
    composition = input_data['composition']
    speed = input_data['speed']
    D = input_data['geometry']['D']
    b = input_data['geometry']['b']

    suc = State.define(p=ps, T=Ts, fluid=composition, **units)

    curve = Curve([
        Point(suc=suc, flow_v=f, head=h, eff=e, speed=speed, **units)
        for f, h, e in zip(flow_v, head, eff)
    ])

    imp = Impeller(curve, b=b, D=D)

    return imp

import os
import csv
import numpy as np
import yaml
from scipy.interpolate import UnivariateSpline
from prf.state import State
from prf.point import Point, Curve
from prf.impeller import Impeller


__all__ = ['load', 'convert_csv_to_yaml']


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


def _interpolated_curve_from_csv(file):
    """Convert from csv file to interpolated curve.

    Function to convert from csv generated with engauge digitizer to an
    interpolated curve.
    """
    flow_values = []
    parameter = []

    with open(file) as csvfile:
        data = csv.reader(csvfile)
        next(data)
        for row in data:
            flow_values.append(float(row[0]))
            parameter.append(float(row[0]))

    parameter_interpolated_curve = UnivariateSpline(flow_values, parameter)
    
    return parameter_interpolated_curve, flow_values


def convert_csv_to_yaml(dir_path='', number_of_points_to_yaml=6):
    """Convert csv head and eff from csv to yaml.
    
    Parameters:
    -----------
    dir_path : str
        Path to directory with head and eff csv files.

    Returns:
    --------
    head_eff : yaml file

    """
    head_file = os.path.join(dir_path, 'head.csv')
    eff_file = os.path.join(dir_path, 'eff.csv')

    head_interpolated_curve, flow_values = _interpolated_curve_from_csv(head_file)
    eff_interpolated_curve, _ = _interpolated_curve_from_csv(eff_file)

    data = {}

    data['flow'] = np.linspace(min(flow_values), max(flow_values),
                               number_of_points_to_yaml)

    data['head'] = head_interpolated_curve(data['flow'])
    data['eff'] = eff_interpolated_curve(data['flow']) / 1e2

    data = {k: [round(float(i), 5) for i in v] for k, v in data.items()}
    print(data)

    yaml_file = os.path.join(dir_path, 'input_head_eff.yml')

    with open(yaml_file, 'w') as f:
        yaml.dump(data, f)
        

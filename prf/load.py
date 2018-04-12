import os
import csv
import numpy as np
import yaml
from scipy.interpolate import UnivariateSpline
from prf.state import State
from prf.point import Point, Curve
from prf.impeller import Impeller


__all__ = ['load', 'load_curve', 'convert_csv_to_dict',
           'convert_head_eff_csv_to_yaml']


def load_curve(file):
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

    suc = State.define(p=ps, T=Ts, fluid=composition, **units)

    curve = Curve([
        Point(suc=suc, flow_v=f, head=h, eff=e, speed=speed, **units)
        for f, h, e in zip(flow_v, head, eff)
    ])

    return curve


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
            parameter.append(float(row[1]))

    parameter_interpolated_curve = UnivariateSpline(flow_values, parameter)
    
    return parameter_interpolated_curve, flow_values


def convert_csv_to_dict(input_path=None, number_of_points=6):
    """Convert csv head and eff from csv to yaml.

    Parameters:
    -----------
    input_path : str
        Path to head csv file.
    output_path : str
        Path to save yaml file.
    number_of_points : int
        Number of points that will be returned from interpolated curve.
    """
    param_interpolated_curve, flow_values = _interpolated_curve_from_csv(input_path)
    flow_range = np.linspace(min(flow_values), max(flow_values),
                             number_of_points)

    # parameter name from input file
    dir_name, file_name = os.path.split(input_path)
    param_name = file_name.split('_')[0]

    data = {'flow': flow_range,
            f'{param_name}': param_interpolated_curve(flow_range)}

    return data


def convert_head_eff_csv_to_yaml(head_path='head.csv', eff_path='eff.csv',
                                 yaml_path='input_head_eff.yml', number_of_points_to_yaml=6):
    """Convert csv head and eff from csv to yaml.
    
    Parameters:
    -----------
    head_path : str
        Path to head csv file.
    eff_path : str
        Path eff csv file.
    yaml_path : str
        Path to save yaml file.

    Returns:
    --------
    head_eff : yaml file

    """

    head_interpolated_curve, flow_values = _interpolated_curve_from_csv(head_path)
    eff_interpolated_curve, _ = _interpolated_curve_from_csv(eff_path)

    flow_range = np.linspace(min(flow_values), max(flow_values),
                             number_of_points_to_yaml)

    data = dict(flow=flow_range,
                head=head_interpolated_curve(flow_range),
                eff=eff_interpolated_curve(flow_range) / 1e2)

    # change np.ndarray to list before dumping to yaml
    data = {k: [round(float(i), 5) for i in v] for k, v in data.items()}

    with open(yaml_path, 'w') as f:
        yaml.dump(data, f)

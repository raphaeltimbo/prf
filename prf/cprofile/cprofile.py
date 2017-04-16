import cProfile
import subprocess
import prf as prf
from datetime import datetime

# current git version / commit
label = subprocess.check_output(['git', 'describe', '--always'])
label = str(label)
date = datetime.now()
date_label = (
    str(date.hour) + 'h' + '-'.join([str(getattr(date, attr)) for attr in ['day', 'month', 'year']]))
if len(label) == 12:
    label = label[2:-3] + '-' + date_label + '.profile'

flow_kgmole_hr = {'Methane': 1964,
                  'Ethane': 135.60,
                  'Propane': 3.85,
                  'n-Butane': 0.1,
                  'Isobutane': 0.1,
                  'Nitrogen': 39.87,
                  'CarbonDioxide': 25.69}

fluid = {}
for k, v in flow_kgmole_hr.items():
    fluid[k] = v/2169.21

units = {
    'p_units': 'kPa',
    'T_units': 'degC',
    'flow_m_units': 'kg/h',
    'flow_v_units': 'm**3/h',
    'speed_units': 'RPM',
    'power_units': 'kW'
}

normal_state = prf.State.define(p=101, T=20, fluid=fluid, **units)
suc = prf.State.define(p=2240.8, T=45, fluid=fluid, **units)
disch = prf.State.define(p=3030.1, T=72.1, fluid=fluid, **units)
flow_m = suc.molar_mass() * 2169
point = prf.Point(suc=suc, disch=disch, flow_m=flow_m, speed=41000, **units)

imp = prf.Impeller(point, b=0.0285, D=0.365)

cProfile.run(
    'point.head_isen(suc=suc, disch=disch)',
    'profile-head_isen-'+label
)

cProfile.run(
    'prf.Point(suc=suc, disch=disch, flow_m=flow_m, speed=41000, **units)',
    'profile-Point-'+label
)

cProfile.run(
    'prf.Impeller(point, b=0.285, D=0.365)',
    'profile-Impeller-'+label
)

fluid = {'Methane': 0.69945,
         'Ethane': 0.09729,
         'Propane': 0.05570,
         'n-Butane': 0.01780,
         'Isobutane': 0.01020,
         'n-Pentane': 0.00390,
         'Isopentane': 0.00360,
         'n-Hexane': 0.00180,
         'Nitrogen': 0.01490,
         'HydrogenSulfide': 0.00017,
         'CarbonDioxide': 0.09259,
         'Water': 0.00200}
units = {'p_units': 'bar',
         'T_units': 'degC',
         'flow_m_units': 'kg/h',
         'speed_units': 'RPM'}

suc, head, eff = (prf.State.define(p=16.99, T=38.4, fluid=fluid, **units),  # suc
                  140349.53763396584,  # head
                  0.71121369651557265)                                  # eff


p0 = prf.Point(suc=suc, head=head, eff=eff, flow_m=175171, speed=12204, **units)

cProfile.run(
    'prf.Point(suc=suc, head=head, eff=eff, flow_m=175171, speed=12204, **units)',
    'profile-Point-from_such_head_eff'+label
)

p1 = prf.Point(suc=p0.suc, eff=p0.eff, volume_ratio=p0.volume_ratio,
               speed=p0.speed, flow_m=p0.flow_m)

cProfile.run(
    'prf.Point(suc=p0.suc, eff=p0.eff, volume_ratio=p0.volume_ratio, speed=p0.speed, flow_m=p0.flow_m)',
    'profile-Point-from_such_eff_vol'+label
)

cProfile.run(
    'prf.Point(suc=point.suc, eff=point.eff, volume_ratio=point.volume_ratio, speed=point.speed, flow_m=point.flow_m)',
    'profile-Point_exp-from_such_eff_vol'+label
)


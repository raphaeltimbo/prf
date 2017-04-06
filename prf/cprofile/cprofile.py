import cProfile
import subprocess
import prf as prf

# current git version / commit
label = subprocess.check_output(['git', 'describe', '--always'])
label = str(label)
if len(label) == 12:
    label = label[2:-3] + '.profile'

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
    'prf.head_isen(suc, disch)',
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

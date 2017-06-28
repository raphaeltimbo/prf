from .impeller import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

__all__ = ['plot_head_curve', 'plot_disch_p_curve', 'plot_eff_curve',
           'plot_envelope', 'plot_power_curve', 'plot_mach']

plt.style.use('seaborn-white')

color_palette = ["#4C72B0", "#55A868", "#C44E52",
                 "#8172B2", "#CCB974", "#64B5CD"]

plt.style.use({
    'lines.linewidth': 2.5,
    'axes.grid': True,
    'axes.linewidth': 0.1,
    'grid.color': '.9',
    'grid.linestyle': '--',
    'legend.frameon': True,
    'legend.framealpha': 0.2
    })

colors = color_palette + [(.1, .1, .1)]
for code, color in zip('bgrmyck', colors):
    rgb = mpl.colors.colorConverter.to_rgb(color)
    mpl.colors.colorConverter.colors[code] = rgb
    mpl.colors.colorConverter.cache[code] = rgb


def plot_head_curve(imp, flow='flow_v', plot_current_point=True,
                    plot_kws=None, ax=None):
    """Plot head curve.
    
    Parameters
    ----------
    imp : prf.Impeller
        Impeller object.
    flow : str, optional
        Flow to be used on x axis. Defaults to 'flow_v' (volumetric).
    plot_current_point : bool, optional
        Plot the impeller current point. Defaults to True.
    plot_kws : dict, optional
        Keyword arguments for underlying plotting functions.
  
    ax : matplotlib.axes, optional
        Matplotlib axes, if None creates a new.
        
    Returns
    -------
    ax : matplotlib.axes
        Matplotlib axes with plot.
        
    """
    if ax is None:
        ax = plt.gca()

    # handle dictionary defaults
    if plot_kws is None:
        plot_kws = dict()

    flow_ = [getattr(p, flow) for p in imp.new_points]

    flow = np.linspace(min(flow_), max(flow_), 100)

    curve, = ax.plot(flow, imp.head_curve(flow), **plot_kws)

    if plot_current_point is True:
        ax.plot(imp.current_point.flow_v, imp.current_point.head, 'o',
                color=curve.get_color())

    if flow is 'flow_m':
        ax.set_xlabel('Mass flow $(kg / s)$')
    else:
        ax.set_xlabel('Volumetric flow $(m^3 / s)$')

    ax.set_ylabel('Head $(J / kg)$')

    return ax


def plot_disch_p_curve(imp, flow='flow_v', plot_current_point=True, ax=None):
    if ax is None:
        ax = plt.gca()

    flow_ = [getattr(p, flow) for p in imp.new_points]

    flow = np.linspace(min(flow_), max(flow_), 100)

    curve, = ax.plot(flow, imp.disch_p_curve(flow))

    if plot_current_point is True:
        ax.plot(imp.current_point.flow_v, imp.current_point.disch.p(), 'o',
                color=curve.get_color())

    if flow is 'flow_m':
        ax.set_xlabel('Mass flow $(kg / s)$')
    else:
        ax.set_xlabel('Volumetric flow $(m^3 / s)$')

    ax.set_ylabel('Discharge Pressure $(Pa)$')

    ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

    return ax


def plot_eff_curve(imp, flow='flow_v', plot_current_point=True, ax=None):
    if ax is None:
        ax = plt.gca()

    flow_ = [getattr(p, flow) for p in imp.new_points]

    flow = np.linspace(min(flow_), max(flow_), 100)

    curve, = ax.plot(flow, imp.eff_curve(flow))

    if plot_current_point is True:
        ax.plot(imp.current_point.flow_v, imp.current_point.eff, 'o',
                color=curve.get_color())

    if flow is 'flow_m':
        ax.set_xlabel('Mass flow $(kg / s)$')
    else:
        ax.set_xlabel('Volumetric flow $(m^3 / s)$')

    ax.set_ylabel('Efficiency')

    return ax


def plot_power_curve(imp, flow='flow_v', plot_current_point=True, ax=None):
    if ax is None:
        ax = plt.gca()

    flow_ = [getattr(p, flow) for p in imp.new_points]

    flow = np.linspace(min(flow_), max(flow_), 100)

    curve, = ax.plot(flow, imp.power_curve(flow))

    if plot_current_point is True:
        ax.plot(imp.current_point.flow_v, imp.current_point.power, 'o',
                color=curve.get_color())

    if flow is 'flow_m':
        ax.set_xlabel('Mass flow $(kg / s)$')
    else:
        ax.set_xlabel('Volumetric flow $(m^3 / s)$')

    ax.set_ylabel('Power $(W)$')

    return ax


def plot_envelope(state, ax=None):
    """Plot phase envelop for a given state.
    
    Parameters
    ----------
    state : prf.State
    
    ax : matplotlib.axes, optional
        Matplotlib axes, if None creates a new.
        
    Returns
    -------
    ax : matplotlib.axes
        Matplotlib axes with plot.
    """
    if ax is None:
        ax = plt.gca()

    state.build_phase_envelope('')
    p_e = state.get_phase_envelope_data()

    ax.plot(p_e.T, p_e.p, '-')

    ax.set_xlabel('Temperature $(K)$')
    ax.set_ylabel('Pressure $(Pa)$')
    ax.set_yscale('log')

    return ax


def plot_mach(imp, ax=None):
    if ax is None:
        ax = plt.gca()

    rng = np.linspace(0, 1.6, 100)

    def limits(mach_sp):
        if mach_sp < 0.214:
            lower_limit = -mach_sp
            upper_limit = -0.25 * mach_sp + 0.286
        elif 0.215 < mach_sp < 0.86:
            lower_limit = 0.266 * mach_sp - 0.271
            upper_limit = -0.25 * mach_sp + 0.286
        else:
            lower_limit = -0.042
            upper_limit = 0.07

        return lower_limit, upper_limit

    lower_limit, upper_limit = [], []
    for i in rng:
        l, u = limits(i)
        lower_limit.append(l)
        upper_limit.append(u)

    curve, = ax.plot(rng, lower_limit)
    ax.plot(rng, upper_limit, color=curve.get_color())
    ax.plot(imp.mach(), imp.new_points[0].mach_comparison['diff'], 'Dr')

    ax.set_xlabel('Mach No. Specified $(Mm_{sp})$')
    ax.set_ylabel('$Mm_t - Mm_{sp}$')

    ax.set_xlim(0, 1.6)

    return ax


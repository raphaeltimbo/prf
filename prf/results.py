from .impeller import *
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

__all__ = ['plot_head_curve', 'plot_disch_p_curve', 'plot_eff_curve',
           'plot_envelope']

sns.set_style('white', rc={'axes.grid':True,
                           'axes.linewidth': 0.1,
                           'grid.color':'.9',
                           'grid.linestyle': '--',
                           'legend.frameon': True,
                           'legend.framealpha': 0.2})
sns.set_context('paper', rc={"lines.linewidth": 1})


def plot_head_curve(imp, flow='flow_v', plot_current_point=True, ax=None):
    if ax is None:
        ax = plt.gca()

    flow_ = [getattr(p, flow) for p in imp.new_points]

    flow = np.linspace(min(flow_), max(flow_), 100)

    ax.plot(flow, imp.head_curve(flow))
    if plot_current_point is True:
        ax.plot(imp.current_point.flow_v, imp.current_point.head, 'o')

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

    ax.plot(flow, imp.disch_p_curve(flow))
    if plot_current_point is True:
        ax.plot(imp.current_point.flow_v, imp.current_point.disch.p(), 'o')

    if flow is 'flow_m':
        ax.set_xlabel('Mass flow $(kg / s)$')
    else:
        ax.set_xlabel('Volumetric flow $(m^3 / s)$')

    ax.set_ylabel('Discharge Pressure $(Pa)$')

    ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))


def plot_eff_curve(imp, flow='flow_v', plot_current_point=True, ax=None):
    if ax is None:
        ax = plt.gca()

    flow_ = [getattr(p, flow) for p in imp.new_points]

    flow = np.linspace(min(flow_), max(flow_), 100)

    ax.plot(flow, imp.eff_curve(flow))
    if plot_current_point is True:
        ax.plot(imp.current_point.flow_v, imp.current_point.eff, 'o')

    if flow is 'flow_m':
        ax.set_xlabel('Mass flow $(kg / s)$')
    else:
        ax.set_xlabel('Volumetric flow $(m^3 / s)$')

    ax.set_ylabel('Efficiency')


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


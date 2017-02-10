from .impeller import *
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

__all__ = ['plot_head_curve']

sns.set_style('white', rc={'axes.grid':True,
                           'axes.linewidth': 0.1,
                           'grid.color':'.9',
                           'grid.linestyle': '--',
                           'legend.frameon': True,
                           'legend.framealpha': 0.2})
sns.set_context('paper', rc={"lines.linewidth": 1})


def plot_head_curve(imp, flow='flow_v', point=None, ax=None):
    if ax is None:
        ax = plt.gca()

    flow_ = [getattr(p, flow) for p in imp.new_points]

    flow = np.linspace(min(flow_), max(flow_), 100)

    ax.plot(flow, imp.head_curve(flow))

    if flow is 'flow_m':
        ax.set_xlabel('Mass flow $(kg / s)$')
    else:
        ax.set_xlabel('Volumetric flow $(m^3 / s)$')

    ax.set_ylabel('Head $(J / kg)$')

    return ax

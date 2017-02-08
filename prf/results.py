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


def plot_head_curve(imp, ax=None):
    if ax is None:
        ax = plt.gca()

    flow_v = [p.flow_v for p in imp.new_points]

    flow = np.linspace(min(flow_v), max(flow_v), 100)

    ax.plot(flow, imp.head_curve(flow))

    return ax

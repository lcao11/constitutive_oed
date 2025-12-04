import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math

def hide_current_axis(*args, **kwds):
    plt.gca().set_visible(False)

def plot_joint_density(data, rowvar=False, labels=None, reference=None, scatter=True, alpha=0.1, color="C0"):
    if rowvar:
        data = np.transpose(data)
    if labels is None:
        labels = [r"$x_{%d}$" % (i) for i in range(data.shape[1])]
    
    df = pd.DataFrame(data, columns=labels)
    # Replace infinite values with NaN to avoid issues with plotting libraries
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # dropna=True handles cases where there are NaNs after replacement
    g = sns.PairGrid(df, dropna=True)
    
    # Pass warn_singular=False to suppress warnings when KDE can't be estimated
    g = g.map_lower(sns.kdeplot, cmap=None, colors="k", linewidths=1, levels=5, common_norm=False, warn_singular=False)
    
    if scatter:
        g = g.map_lower(sns.scatterplot, s=5, linewidth=0, alpha=alpha, color=color)
    else:
        g = g.map_lower(sns.histplot, color=color, stat="density")
    g = g.map_diag(sns.histplot, edgecolor="black", color="k", element="step",                 fill=True, alpha=0.0)
    g = g.map_upper(hide_current_axis)

    idx = 0
    ax = g.axes.flat
    size = int(np.sqrt(len(ax)))
    for row in range(size):
        for col in range(size):
            if col <= row:
                if not (reference is None):
                    ax[idx].axvline(x=reference[col], linestyle="--", color="r")
                ax[idx].spines['top'].set_visible(True)
                ax[idx].spines['right'].set_visible(True)
            if col < row:
                if not (reference is None):
                    ax[idx].axhline(y=reference[row], linestyle="--", color="r")
            ax[idx].tick_params(axis='x', labelrotation=30)
            ax[idx].tick_params(axis='y', labelrotation=30)
            idx += 1
    ax[0].set_yticks([])
    ax[0].set_ylabel("")
    return g

def plot(data, rowvar=False, hue=None, reference=None, palette=None):
    if rowvar:
        data = np.transpose(data)
    assert hue is not None
    df = pd.DataFrame(data)
    count = len(list(set(df[hue])))
    g = sns.PairGrid(df, hue=hue, palette=palette)
    g._legend_out = False
    g = g.map_lower(sns.kdeplot, linewidths=1, levels=5, alpha=1.0, common_norm=False)
    g = g.map_diag(sns.histplot, edgecolor="None", stat="probability", element="step", \
                   fill=True, alpha=1. / count)
    g = g.map_upper(hide_current_axis)

    idx = 0
    ax = g.axes.flat
    size = int(np.sqrt(len(ax)))
    for row in range(size):
        for col in range(size):
            if col <= row:
                if not (reference is None):
                    ax[idx].axvline(x=reference[col], linestyle="--", color="r")
                ax[idx].spines['top'].set_visible(True)
                ax[idx].spines['right'].set_visible(True)
            if col < row:
                if not (reference is None):
                    ax[idx].axhline(y=reference[row], linestyle="--", color="r")
            ax[idx].tick_params(axis='x', labelrotation=30)
            ax[idx].tick_params(axis='y', labelrotation=30)
            idx += 1
    ax[0].set_yticks([])
    ax[0].set_ylabel("")

    return g

def plot_multiple_joint_density(data_list, name_list, sample_labels=None, prior_contour=False, palette=None, reference=None):
    assert len(data_list) == len(name_list)
    data_array = np.concatenate(data_list, axis=0)
    hue_list = []
    for i, data in enumerate(data_list):
        for j in range(data.shape[0]):
            hue_list.append(name_list[i])

    data_frame = {}
    for i in range(data_array.shape[1]):
        if sample_labels is None:
            data_frame["$x_{%d}$" % (i)] = data_array[:, i]
        else:
            data_frame[sample_labels[i]] = data_array[:, i]
    data_frame["type"] = hue_list
    
    if palette is None:
        palette = dict(zip(name_list, ["C" + str(i) for i in range(len(name_list))]))
    g = plot(data_frame, hue="type", palette=palette, reference=reference)

    if prior_contour:
        delta = 0.025
        x = np.arange(-3.0, 3.0, delta)
        y = np.arange(-3.0, 3.0, delta)
        X, Y = np.meshgrid(x, y)
        Z = (1.0 / (2 * math.pi)) * np.exp(-0.5 * X ** 2 - 0.5 * Y ** 2)
        idx = 0
        ax = g.axes.flat
        size = int(np.sqrt(len(ax)))
        legend_flag = True
        for row in range(size):
            for col in range(size):
                if col < row:
                    if legend_flag:
                        ax[idx].contour(X, Y, Z, 5, colors='k')
                        legend_flag = False
                    else:
                        ax[idx].contour(X, Y, Z, 5, colors='k')
                ax[idx].set_xlim((-3.0, 3.0))
                ax[idx].set_ylim((-3.0, 3.0))
                idx += 1
        ax[0].set_yticks([])
        ax[0].set_ylabel("")
    g._legend_out = False
    return g
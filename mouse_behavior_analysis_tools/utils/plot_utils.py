# plot_utils
import math
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from numpy.core.numeric import NaN
from scipy import stats

import mouse_behavior_analysis_tools.utils.custom_functions as cuf


def axvlines(xs, ax=None, **plot_kwargs):
    """
    Function from StackExchange
    Draw vertical lines on plot
    :param xs: A scalar, list, or 1D array of horizontal offsets
    :param ax: The axis (or none to use gca)
    :param plot_kwargs: Keyword arguments to be passed to plot
    :return: The plot object corresponding to the lines.
    """
    if ax is None:
        ax = plt.gca()
    xs = np.array((xs,) if np.isscalar(xs) else xs, copy=False)
    lims = ax.get_ylim()
    x_points = np.repeat(xs[:, None], repeats=3, axis=1).flatten()
    y_points = np.repeat(
        np.array(lims + (np.nan,))[None, :], repeats=len(xs), axis=0
    ).flatten()
    plot = ax.plot(x_points, y_points, scaley=False, **plot_kwargs)
    return plot


def summary_figure_joystick(mydict, subplot_time_length=300):
    # generate a summary figure of the training
    # calculate the number of subplots needed
    # subplot_time_length = 300  # 5 minutes
    init_time = mydict["Moving_az"]["MovingAzimuthTimes"][0]
    final_time = mydict["Moving_az"]["MovingAzimuthTimes"][
        mydict["Moving_az"].shape[0] - 1
    ]
    duration = final_time - init_time
    number_of_subplots = int(math.floor(duration / subplot_time_length + 1))

    fig, axs = plt.subplots(
        number_of_subplots,
        1,
        sharey=False,
        sharex=False,
        figsize=(18, 3 * number_of_subplots),
        dpi=80,
        facecolor="w",
        edgecolor="k",
    )
    for i in range(0, number_of_subplots):
        summary_plot(mydict, ax=axs[i])
        axs[i].set_xlim(
            init_time + i * subplot_time_length,
            init_time + (i + 1) * subplot_time_length,
        )
        axs[i].set_ylim(-50, 60)
    axs[i].legend()
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.suptitle(mydict["Main_name"] + "_summary", size=24)
    fig.text(0.5, 0, "Time", ha="center")
    fig.text(0, 0.5, "Moving Azimuth", va="center", rotation="vertical")

    return fig


def summary_plot_joystick(mydict, ax=None):
    """
    Create a summary plot with info about the training session
    :param mydict: dictionary containing the data
    :param ax: axis of the plot
    :return: axis with plot data
    """
    if ax is None:
        ax = plt.gca()
    ax.plot(
        "MovingAzimuthTimes",
        "MovingAzimuthValues",
        data=mydict["Moving_az"],
        color="blue",
        linewidth=1,
    )
    ax.plot(
        "TrialSideTimes",
        "TrialSideVM",
        data=mydict["Trial_side"],
        color="grey",
        marker=".",
        linewidth=0.5,
    )
    ax.plot(
        mydict["Target_reached"],
        len(mydict["Target_reached"]) * [50],
        ".",
        color="green",
        alpha=0.5,
    )
    ax.plot(
        mydict["Wrong_reached"],
        len(mydict["Wrong_reached"]) * [50],
        ".",
        color="red",
        alpha=0.5,
    )
    axvlines(mydict["Lick_events"], ax=ax, linewidth=0.2, color="gray")

    return ax


def PlotPsychPerformance(
    dataDif=None,
    dataPerf=None,
    predictDif=None,
    ax=None,
    realPred=None,
    fakePred=None,
    errorBars=None,
    label="data",
    **plot_kwargs,
):
    # Plots different various features of the psychometric performance

    if ax is None:
        ax = plt.gca()

    # This plots all the fake data:
    # plt.plot(predictDif, fakePred, 'k-', lw=0.5, alpha=0.2)

    # plot percentiles if fake data is provided
    if fakePred is not None:
        percentiles = np.percentile(fakePred, [2.5, 97.5], axis=1).T
        ax.fill_between(
            predictDif.reshape(-1),
            percentiles[:, 0],
            percentiles[:, 1],
            alpha=0.2,
            **plot_kwargs,
        )

    # plot the psychometric performance if the predictions are provided
    if realPred is not None:
        ax.plot(predictDif.reshape(-1), realPred, "-", **plot_kwargs)

    # plot the error bars
    if errorBars is not None:
        for i, EBlength in enumerate(errorBars):
            ax.plot(
                [dataDif[i], dataDif[i]],
                [dataPerf[i] - EBlength / 2, dataPerf[i] + EBlength / 2],
                "-",
                **plot_kwargs,
            )

    # plot the data
    if dataPerf is not None:
        ax.plot(dataDif, dataPerf, "o", ms=8, label=label, **plot_kwargs)

    # make the plot pretty
    if dataDif is not None:
        ax.set_xticks(dataDif)
    ax.set_ylabel("% Rightward choices")
    ax.set_xlabel("% High tones")
    ax.set_xlim(0.0, 100.0)
    ax.set_ylim(-2.0, 102.0)
    ax.legend(bbox_to_anchor=(1.05, 1), loc=0, borderaxespad=0.0)
    ax.tick_params(
        which="both",
        top="off",
        bottom="on",
        left="on",
        right="off",
        labelleft="on",
        labelbottom="on",
    )
    # get rid of the frame
    for spine in ax.spines.values():
        spine.set_visible(False)

    # invert the axis as it looks more natural for a psychometric curve
    ax.invert_xaxis()

    return ax


def summary_matrix(df):
    # Initialize lists to save important data
    DifficultyValues = []
    PerformanceValues = []

    # process data from all experiments
    for counter, session in enumerate(pd.unique(df["SessionTime"])):
        (
            predictDif,
            PsyPer,
            fakePredictions,
            predictPer,
            _,
        ) = cuf.PP_ProcessExperiment(df[df["SessionTime"] == session])

        # append to lists, only the normal trials
        DifficultyValues.append(PsyPer["Difficulty"])
        PerformanceValues.append(PsyPer["Performance"])

        # OE.update_progress(counter / len(pd.unique(df['SessionTime'])))

    # OE.update_progress(1)

    # calculate difficulty levels
    difLevels = np.unique(np.concatenate(DifficultyValues).ravel())
    # Initialize the matrix
    matToPlot = np.full([len(difLevels), len(DifficultyValues)], np.nan)
    # Loop to fill it
    for i, dif in enumerate(difLevels):
        for j, per in enumerate(PerformanceValues):
            if dif in DifficultyValues[j]:
                idxOfDif = np.where(DifficultyValues[j] == dif)[0][0]
                matToPlot[i, j] = per[idxOfDif]

    # Transform to dataframe
    dfToPlot = pd.DataFrame(matToPlot)
    dfToPlot = dfToPlot.set_index(difLevels)  # set row names
    dfToPlot.columns = pd.unique(df["SessionTime"])  # set col names

    return dfToPlot


def summary_plot(
    dfToPlot, AnimalDF, ax, top_labels=["Stimulation", "Muscimol"]
):
    """
    Generates a matrix plot with information regarding
    each particular session on the top
    """
    sns.set(style="white")
    sp = sns.heatmap(
        dfToPlot,
        linewidth=0.001,
        square=True,
        cmap="coolwarm",
        cbar_kws={"shrink": 0.6, "label": "% Rightward choices"},
        ax=ax,
        vmin=0,
        vmax=100,
    )
    # TODO: check that the size is proportional (area vs radius)
    # recalculate the number of trials as some might
    # get grouped if they are on the same day.
    # Do all below with the dataframe

    # The protocols is the default that gets plotted
    Protocols = [
        pd.unique(AnimalDF[AnimalDF["SessionTime"] == session]["Protocol"])[0]
        for session in pd.unique(AnimalDF["SessionTime"])
    ]
    ntrialsDistribution = [
        len(AnimalDF[AnimalDF["SessionTime"] == session])
        for session in pd.unique(AnimalDF["SessionTime"])
    ]

    difLevels = dfToPlot.index
    AnimalName = str(pd.unique(AnimalDF.AnimalID)[0])
    AnimalGroup = str(pd.unique(AnimalDF.ExperimentalGroup)[0])
    shift_up = 0.5

    for pr_counter, prot in enumerate(np.unique(Protocols)):
        protIdx = [i for i, x in enumerate(Protocols) if x == prot]
        ax.scatter(
            [x + 0.5 for x in protIdx],
            np.repeat(len(difLevels) + shift_up, len(protIdx)),
            marker="o",
            s=[ntrialsDistribution[x] / 5 for x in protIdx],
            label=prot,
        )
    shift_up += 1

    # label the rest of teh seesions as given in the input
    marker_list = ["*", "P", "h", "D", "X"]
    for n_lab, t_label in enumerate(top_labels):
        t_label_uniques = [
            pd.unique(AnimalDF[AnimalDF["SessionTime"] == session][t_label])[0]
            for session in pd.unique(AnimalDF["SessionTime"])
        ]
        for st_counter, tlu in enumerate(np.unique(t_label_uniques)):
            tlu_idx = [i for i, x in enumerate(t_label_uniques) if x == tlu]
            ax.scatter(
                [x + 0.5 for x in tlu_idx],
                np.repeat(len(difLevels) + shift_up, len(tlu_idx)),
                marker=marker_list[n_lab],
                s=100,
                label=tlu,
            )
        shift_up += 1

    ax.legend(loc=(0, 1), borderaxespad=0.0, ncol=5, frameon=True)
    ax.set_ylim([0, len(difLevels) + shift_up - 0.5])
    plt.ylabel("% High Tones")
    plt.xlabel("Session")
    sp.set_yticklabels(sp.get_yticklabels(), rotation=0)
    sp.set_xticklabels(
        sp.get_xticklabels(), rotation=45, horizontalalignment="right"
    )
    sp.set_title(
        AnimalName + " - " + AnimalGroup + "\n\n", fontsize=20, fontweight=0
    )

    return ax


def reg_in_ax(X, Y, ax, legloc):
    # plots a regression in the axes

    # solution with sklearn, deprecated as pvalues
    # are easy to calculate with stats
    """
    # fit model
    regr = LinearRegression()
    lrmodel = regr.fit(X[:, np.newaxis], Y)
    # predict
    Xpr = np.linspace(np.min(X), np.max(X), 30)
    Ypr = regr.predict(Xpr[:, np.newaxis])
    # r score:
    r_sq = regr.score(X[:, np.newaxis], Y)
    """
    # solution with stats
    slope, intercept, r_value, p_value, std_err = stats.linregress(X, Y)
    Xpr = np.linspace(np.min(X), np.max(X), 30)
    Ypr = Xpr * slope + intercept
    r_sq = r_value**2
    # plot
    ax.plot(Xpr, Ypr, color="k", linestyle="--")
    legend_elements = [
        Line2D(
            [0],
            [0],
            color="k",
            lw=1,
            linestyle="--",
            label="r2 = %.3f\np-val = %.3f" % (r_sq, p_value),
        )
    ]
    ax.legend(handles=legend_elements, loc=legloc, frameon=False)


def add_stats(ax=None, xvals=(0, 1), yval=None, pval=None, n_asterisks=1):
    # get axis
    if ax is None:
        ax = plt.gca()
    # if yval is None plot in the middle
    ylims = ax.get_ylim()
    if yval is None:
        yval = (ylims[1] + ylims[0]) / 2
    # plot two lines with n_asterisks in between
    # each asterisk occupies a certain % of the xvals specified
    # define that middle space
    xlength = xvals[1] - xvals[0]
    xmidpoint = (xvals[1] + xvals[0]) / 2
    mid_space = n_asterisks * 0.2 * xlength
    # define where the first line end and where the second begin
    first_end = xmidpoint - mid_space / 2
    second_begin = first_end + mid_space

    # plot the horizontal bars
    ax.plot([xvals[0], first_end], [yval, yval], "gray")
    ax.plot([second_begin, xvals[1]], [yval, yval], "gray")

    # add asterisks
    ast_str = n_asterisks * "*"
    ax.text(
        xmidpoint,
        yval,
        ast_str,
        horizontalalignment="center",
        verticalalignment="center",
    )

    # add small vertical bars
    vbsize = 0.05 * (ylims[1] - ylims[0])
    ax.plot([xvals[0], xvals[0]], [yval, yval + vbsize], "gray")
    ax.plot([xvals[1], xvals[1]], [yval, yval + vbsize], "gray")


def plot_regression(df, ax, color, label, plot_points=True):

    if ax is None:
        ax = plt.gca()

    trialsDif = np.array(df["TrialHighPerc"])
    sideSelected = np.array(df["FirstPoke"])

    difficulty, performance = cuf.get_choices(sideSelected, trialsDif)

    slope, bias, upper_lapse, lower_lapse = cuf.fit_custom_sigmoid(
        difficulty, performance
    )

    x = np.linspace(0, 100)

    if plot_points:
        ax.plot(difficulty, performance, "o", ms=8, color=color, label=label)

    sns.lineplot(
        x=x,
        y=cuf.sigmoid_func(x, *[slope, bias, upper_lapse, lower_lapse]),
        color=color,
        ci=None,
        ax=ax,
        label=label,
    )

    return ax


def plot_random_optolike_choices(
    df,
    ax,
    fake_dataset_m_and_std=[NaN, NaN, NaN],
    normalize=False,
    jitter=0,
    colorlist=None,
):

    if colorlist is None:
        colorlist = ["c", "m"]

    imp_jit = random.uniform(-jitter, jitter)

    if ax is None:
        ax = plt.gca()

    normal_df, opto_df = cuf.splitOpto(df)

    difficulty_n, choice_n = cuf.get_choices(
        np.array(normal_df["SideSelected"]), np.array(normal_df["Difficulty"])
    )
    difficulty_o, choice_o = cuf.get_choices(
        np.array(opto_df["SideSelected"]), np.array(opto_df["Difficulty"])
    )

    if normalize:
        # check here that there are opto trials for each difficulty
        norm_mask = np.where(np.in1d(difficulty_n, difficulty_o))[0]

        choice_n = choice_n[norm_mask] - choice_o

    if not normalize:
        ax.plot(
            difficulty_n + imp_jit, choice_n, "o", ms=10, color=colorlist[0]
        )

    tr_means = fake_dataset_m_and_std[1]
    tr_dif = fake_dataset_m_and_std[0]
    if normalize:
        tr_means = tr_means[norm_mask] - choice_o
        tr_dif = tr_dif[norm_mask]

    ax.plot(tr_dif + imp_jit, tr_means, "o", color=colorlist[0], ms=8)

    for dif in fake_dataset_m_and_std[0]:
        dif_idx = np.where(fake_dataset_m_and_std[0] == dif)
        m = np.float(fake_dataset_m_and_std[1][dif_idx])
        s = np.float(fake_dataset_m_and_std[2][dif_idx])

        if normalize:
            dif_index = np.where(difficulty_o == dif)[0]
            if dif_index.size > 0:
                m = m - choice_o[dif_index]
                ax.plot(
                    [dif + imp_jit, dif + imp_jit],
                    [m - s, m + s],
                    "-",
                    color=colorlist[0],
                )

        else:
            ax.plot(
                [dif + imp_jit, dif + imp_jit],
                [m - s, m + s],
                "-",
                color=colorlist[0],
            )

    if not normalize:
        ax.plot(difficulty_o, choice_o, "o", ms=8, color=colorlist[1])

    return ax


def plot_trials_over_learning(
    ax, data, line_to_add=False, axtitle=False, override_hue=False
):

    ax.hlines(50, 0, 5000, linestyles="dotted", alpha=0.4)

    # plot here
    if override_hue:
        sns.scatterplot(
            data=data,
            x="CumulativeTrialNumberByProtocol",
            y="CurrentPastPerformance100",
            marker=".",
            color=override_hue,
            alpha=0.1,
            ax=ax,
        )

    else:
        sns.scatterplot(
            data=data,
            x="CumulativeTrialNumberByProtocol",
            y="CurrentPastPerformance100",
            marker=".",
            hue="SessionID",
            alpha=0.1,
            ax=ax,
        )

    # plot a line for binned trials
    if line_to_add:
        sns.lineplot(
            x=line_to_add[0], y=line_to_add[1], color="k", ci=None, ax=ax
        )

    if axtitle:
        ax.text(
            0.5,
            0.95,
            axtitle,
            horizontalalignment="center",
            fontweight="bold",
            transform=ax.transAxes,
        )

    ax.axis("on")


def plot_swarm_and_boxplot(fit_df, var, ax, hue_order, spread, color_palette):
    sns.swarmplot(
        data=fit_df,
        x="ExperimentalGroup",
        y=var,
        order=hue_order,
        hue="ExperimentalGroup",
        hue_order=hue_order,
        dodge=False,  # jitter=.25,
        alpha=0.5,
        zorder=1,
        size=8,
        ax=ax,
    )

    # boxplot next to it
    for k, egr in enumerate(hue_order):
        bpdat = fit_df[fit_df.ExperimentalGroup == egr][var].values
        bp = ax.boxplot(
            [bpdat],
            positions=[k + spread],
            widths=0.1,
            patch_artist=True,
            showfliers=False,
        )
        for element in [
            "boxes",
            "whiskers",
            "fliers",
            "means",
            "medians",
            "caps",
        ]:
            plt.setp(bp[element], color=color_palette[k], linewidth=2)
        for patch in bp["boxes"]:
            patch.set(facecolor="white")

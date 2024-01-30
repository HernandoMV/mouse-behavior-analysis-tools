import math
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from mouse_behavior_analysis_tools.utils import custom_functions as cuf
from mouse_behavior_analysis_tools.utils import model_utils, plot_utils
from mouse_behavior_analysis_tools.utils.misc_utils import update_progress


def make_figure_performance_trials_animals_bin(df_to_plot):

    ans_list = np.sort(df_to_plot.AnimalID.unique())
    num_ans = len(ans_list)
    fig, axs = plt.subplots(
        math.ceil(num_ans / 3),
        3,
        figsize=(15, num_ans),
        facecolor="w",
        edgecolor="k",
        sharey=True,
        sharex=True,
    )
    fig.subplots_adjust(hspace=0.2, wspace=0.2)
    axs = axs.ravel()
    for ax in axs:
        ax.axis("off")
    # process data from all animals
    for counter, animal in enumerate(ans_list):
        ax = axs[counter]
        data = df_to_plot[df_to_plot.AnimalID == animal]
        linetp = [
            df_to_plot[df_to_plot.AnimalID == animal]["TrialIndexBinned200"]
            + 100,
            # trick to align as CurrentPastPerformance looks at the past
            100
            * df_to_plot[df_to_plot.AnimalID == animal]["FirstPokeCorrect"],
        ]

        ec = df_to_plot[
            df_to_plot.AnimalID == animal
        ].ExperimentalGroup.unique()[0]
        ax_title = ec + ": " + animal

        plot_utils.plot_trials_over_learning(ax, data, linetp, ax_title)
        ax.get_legend().remove()

        plt.tight_layout()

        update_progress(counter / num_ans)

    update_progress(1)

    return fig


def make_figure_performance_trials_animals_model(
    df_to_plot, fit_df, der_max_dir
):

    ans_list = np.sort(df_to_plot.AnimalID.unique())
    num_ans = len(ans_list)
    x = np.linspace(1, 5000)

    fig, axs = plt.subplots(
        math.ceil(num_ans / 3),
        3,
        figsize=(15, num_ans),
        facecolor="w",
        edgecolor="k",
        sharey=True,
        sharex=True,
    )
    fig.subplots_adjust(hspace=0.2, wspace=0.2)
    axs = axs.ravel()
    for ax in axs:
        ax.axis("off")
    # process data from all animals
    for counter, animal in enumerate(ans_list):
        ax = axs[counter]
        data = df_to_plot[df_to_plot.AnimalID == animal][
            [
                "CumulativeTrialNumberByProtocol",
                "CurrentPastPerformance100",
                "SessionID",
            ]
        ].dropna()
        linetp = [
            x,
            model_utils.sigmoid_func_sc(
                x,
                *[
                    fit_df[fit_df.AnimalID == animal].maximum_performance.iloc[
                        0
                    ],
                    fit_df[fit_df.AnimalID == animal].slope.iloc[0],
                    fit_df[fit_df.AnimalID == animal].bias.iloc[0],
                ],
            ),
        ]

        ec = df_to_plot[
            df_to_plot.AnimalID == animal
        ].ExperimentalGroup.unique()[0]
        ax_title = ec + ": " + animal

        plot_utils.plot_trials_over_learning(ax, data, linetp, ax_title)
        ax.get_legend().remove()

        # point to the maximum slope
    for counter, animal in enumerate(ans_list):
        ax = axs[counter]
        ymin, ymax = ax.get_ybound()
        perc_max_slope = model_utils.sigmoid_func_sc(
            der_max_dir[animal][0],
            *[
                fit_df[fit_df.AnimalID == animal].maximum_performance.iloc[0],
                fit_df[fit_df.AnimalID == animal].slope.iloc[0],
                fit_df[fit_df.AnimalID == animal].bias.iloc[0],
            ],
        )
        ax.axvline(
            fit_df[fit_df.AnimalID == animal].bias.iloc[0],
            0,
            (perc_max_slope - ymin) / (ymax - ymin),
            linestyle="--",
            color="k",
        )
        ax.plot(
            [0, fit_df[fit_df.AnimalID == animal].bias.iloc[0]],
            [perc_max_slope, perc_max_slope],
            "k--",
        )

        plt.tight_layout()

        update_progress(counter / num_ans)

    update_progress(1)
    return fig


def make_figure_learning_parameters_between_groups(
    fit_df,
    parameters_to_show,
    titles,
    ylabs,
    pvals,
    sig_levels,
    color_palette,
    hue_order,
    yvals,
):

    spread = 0.3

    fig, axs = plt.subplots(
        ncols=len(parameters_to_show), nrows=1, sharey=False, figsize=(6, 4)
    )
    axs = axs.ravel()
    for i, var in enumerate(parameters_to_show):

        plot_utils.plot_swarm_and_boxplot(
            fit_df, var, axs[i], hue_order, spread, color_palette
        )

        axs[i].set_title(titles[i])
        axs[i].set_ylabel(ylabs[i])

    for ax in axs:

        # Hide the right and top spines
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)
        # Only show ticks on the left
        # ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks([])
        ax.xaxis.set_visible(False)
        ax.get_legend().remove()

    # add statistics
    for i, ax in enumerate(axs):
        n_ast = sum(pvals[i] < sig_levels)
        plot_utils.add_stats(
            ax,
            xvals=(spread / 2, 1 + spread / 2),
            yval=yvals[i],
            n_asterisks=n_ast,
        )

    plt.tight_layout()

    return fig


def make_figure_performance_trials_animals_biased_trials(
    df_to_plot, bias_mask
):

    ans_list = np.sort(df_to_plot.AnimalID.unique())
    num_ans = len(ans_list)
    fig, axs = plt.subplots(
        math.ceil(num_ans / 3),
        3,
        figsize=(15, num_ans),
        facecolor="w",
        edgecolor="k",
        sharey=True,
        sharex=True,
    )
    fig.subplots_adjust(hspace=0.2, wspace=0.2)
    axs = axs.ravel()
    for ax in axs:
        ax.axis("off")
    # process data from all animals
    for counter, animal in enumerate(ans_list):
        ax = axs[counter]
        data = df_to_plot[df_to_plot.AnimalID == animal]

        ec = df_to_plot[
            df_to_plot.AnimalID == animal
        ].ExperimentalGroup.unique()[0]
        ax_title = ec + ": " + animal

        plot_utils.plot_trials_over_learning(
            ax, data, line_to_add=False, axtitle=ax_title, override_hue="red"
        )
        plot_utils.plot_trials_over_learning(
            ax,
            data[bias_mask],
            line_to_add=False,
            axtitle=False,
            override_hue="green",
        )

        plt.tight_layout()

        update_progress(counter / num_ans)

    update_progress(1)

    return fig


def make_figure_differences_performance_between_groups(
    df_to_plot, col_to_plot, hue_order, color_palette
):
    data_mean = (
        df_to_plot.groupby(
            ["CumulativeTrialNumberByProtocol", "ExperimentalGroup"]
        )[col_to_plot]
        .mean()
        .reset_index()
    )
    st_err_mean = (
        df_to_plot.groupby(
            ["CumulativeTrialNumberByProtocol", "ExperimentalGroup"]
        )[col_to_plot]
        .std()
        .reset_index()
    )
    data_mean["low_bound"] = data_mean[col_to_plot] - st_err_mean[col_to_plot]
    data_mean["high_bound"] = data_mean[col_to_plot] + st_err_mean[col_to_plot]

    fig1 = plt.figure(figsize=(8, 4))
    plt.axhline(50, ls="dotted", alpha=0.4, color="k")
    plt.axhline(100, ls="dotted", alpha=0.4, color="k")
    for i, eg in enumerate(hue_order):
        df = data_mean[data_mean.ExperimentalGroup == eg].copy()
        x = df.CumulativeTrialNumberByProtocol
        plt.plot(x, df[col_to_plot], color=color_palette[i], label=eg)
        y1 = df["low_bound"]
        y2 = df["high_bound"]
        plt.fill_between(
            x,
            y1,
            y2,
            where=y2 >= y1,
            color=color_palette[i],
            alpha=0.2,
            interpolate=False,
        )

    plt.ylabel(col_to_plot)
    plt.xlabel("trial number")
    plt.ylabel("task performance (%)")
    plt.legend(loc=(0.76, 0.3), frameon=False)

    ax = plt.gca()
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    # remove the legend as the figure has it's own
    ax.get_legend().remove()

    ax.set_xlim((0, 5000))

    plt.title("Task learning progression")

    return fig1


def make_figure_differences_performance_significance(
    real_data_pd, pos_ci, neg_ci
):
    fig2 = plt.figure(figsize=(8, 4))
    plt.axhline(0, ls="dotted", alpha=0.4, color="k")
    plt.plot(real_data_pd, color="k", label="observed data")
    plt.plot(pos_ci, linestyle="--", color="gray", label="95% ci")
    plt.plot(neg_ci, linestyle="--", color="gray")
    x = pos_ci.reset_index().TrialIndexBinned
    y1 = neg_ci.reset_index().Performance
    y2 = real_data_pd.reset_index().Performance
    plt.fill_between(
        x, y1, y2, where=y1 >= y2, facecolor="k", alpha=0.2, interpolate=True
    )
    plt.ylabel("performance difference (%)")
    plt.xlabel("trial number")
    plt.legend(loc=(0.75, 0.05), frameon=False)
    ax = plt.gca()
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.set_xlim((0, 5000))

    return fig2


def make_figure_differences_performance_significance_global(
    real_data_pd, quants_to_test, shrdf, global_sig, nsh
):
    fig = plt.figure(figsize=(16, 4))
    sns.lineplot(data=real_data_pd, color="r")
    for k, q in enumerate(quants_to_test):
        sns.lineplot(
            data=shrdf.groupby("TrialIndexBinned").quantile(q), color="k"
        )
        sns.lineplot(
            data=shrdf.groupby("TrialIndexBinned").quantile((1 - q)), color="k"
        )
        print(
            "ci = ",
            q,
            "\tglobal pval = ",
            np.sum(global_sig, axis=0)[k] / nsh,
            "\treal data significant ",
            any(
                np.logical_or(
                    real_data_pd
                    > shrdf.groupby("TrialIndexBinned").quantile(q),
                    real_data_pd
                    < shrdf.groupby("TrialIndexBinned").quantile(1 - q),
                )
            ),
        )

    return fig


def make_figure_muscimol_sessions_overview(mus_df):
    # plot a summary of all the animals in the dataset
    fig, ax = plt.subplots(
        len(pd.unique(mus_df.AnimalID)),
        1,
        figsize=(7, 5 * len(pd.unique(mus_df.AnimalID))),
    )
    axs = ax.ravel()
    fig.subplots_adjust(hspace=1.3)
    for i, animal in enumerate(pd.unique(mus_df.AnimalID)):
        aDF = mus_df[mus_df.AnimalID == animal]
        dfToPlot = plot_utils.summary_matrix(aDF)
        axs[i] = plot_utils.summary_plot(
            dfToPlot, aDF, axs[i], top_labels=["Muscimol"]
        )

    return fig


def make_figure_muscimol_psychometric(PP_array, muscond_text, colorlist):
    fig = plt.figure(figsize=(5, 5), facecolor="w", edgecolor="k")
    ax = plt.gca()
    ax.hlines(50, 0, 100, linestyles="dotted", alpha=0.4)

    for counter, results in enumerate(PP_array):
        predictDif, PsyPer, fakePredictions, predictPer, EB = results
        plot_utils.PlotPsychPerformance(
            dataDif=PsyPer["Difficulty"],
            dataPerf=PsyPer["Performance"],
            predictDif=predictDif,
            ax=ax,
            fakePred=fakePredictions,
            realPred=predictPer,
            label=muscond_text[counter],
            errorBars=EB,
            color=colorlist[counter],
        )
    ax.axis("on")
    # remove some ticks
    ax.tick_params(
        which="both",
        top=False,
        bottom="on",
        left="on",
        right=False,
        labelleft="on",
        labelbottom="on",
    )

    plt.legend(loc="upper left", frameon=False)
    # L.get_texts()[0].set_text('Saline (str tail)')
    # L.get_texts()[1].set_text('Muscimol (str tail)')
    # L.get_texts()[2].set_text('Muscimol (DMS)')

    return fig


def make_figure_optoinhibition_after_learning_batch(random_opto_df):

    jitter = 0.3
    alpha = 1
    spread = jitter * 1.6
    mice_cohorts = ["D1opto", "D2opto"]
    colors = ["skyblue", "olivedrab"]
    labels_for_legend = ["D1-Arch", "D2-Arch"]

    fig, axs = plt.subplots(
        1, len(mice_cohorts), figsize=(4 * len(mice_cohorts), 8), sharey=True
    )

    axs = axs.ravel()
    for i, ax in enumerate(axs):
        ax.axhline(0, color="grey", linestyle="--")
        ax.set_title(labels_for_legend[i], fontsize=20)
        ax.set_xticks([])
        ax.set_xlim([-jitter * 1.2, jitter * 3])
        # get rid of the frame
        for spine in ax.spines.values():
            spine.set_visible(False)

        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(14)

        xmin, _ = axs[0].get_xaxis().get_view_interval()
        ax.plot((xmin, xmin), (-60, 40), color="black", linewidth=1)

    axs[0].set_ylabel("contralateral bias", fontsize=15)

    jit_list = []

    # plot stds
    for session in pd.unique(random_opto_df.SessionID):
        session_idx = random_opto_df.index[
            random_opto_df.SessionID == session
        ].item()
        cohort = random_opto_df.loc[session_idx].Genotype
        ax = axs[mice_cohorts.index(cohort)]
        st_t_idx = 0
        sh_d = random_opto_df.loc[session_idx].contralateral_bias
        sh_std = random_opto_df.loc[session_idx].bias_std
        imp_jit = random.uniform(-jitter, jitter)
        x_pos = st_t_idx + imp_jit
        jit_list.append(x_pos)

        # stds
        ax.plot(
            [x_pos, x_pos],
            [sh_d - sh_std, sh_d + sh_std],
            color=colors[mice_cohorts.index(cohort)],
            linewidth=3,
            alpha=alpha,
        )

    counter = 0
    # plot means on top
    mean_vals = [[], []]
    sessions_used = [[], []]
    for session in pd.unique(random_opto_df.SessionID):
        session_idx = random_opto_df.index[
            random_opto_df.SessionID == session
        ].item()
        cohort = random_opto_df.loc[session_idx].Genotype
        ax = axs[mice_cohorts.index(cohort)]
        st_t_idx = 0
        sh_d = random_opto_df.loc[session_idx].contralateral_bias
        imp_jit = random.uniform(-jitter, jitter)
        x_pos = jit_list[counter]
        counter += 1

        # means
        ax.plot(
            x_pos,
            sh_d,
            "o",
            ms=14,
            color="k",
            markerfacecolor=colors[mice_cohorts.index(cohort)],
        )
        # append to list
        mean_vals[mice_cohorts.index(cohort)].append(sh_d)
        sessions_used[mice_cohorts.index(cohort)].append(session)

    # plot mean of means next to it, and random distribution, and pvalue
    pvals = []
    for i, ax in enumerate(axs):
        bp = ax.boxplot(
            mean_vals[i],
            positions=[spread],
            widths=0.07,
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
            plt.setp(bp[element], color=colors[i], linewidth=3)
        for patch in bp["boxes"]:
            patch.set(facecolor="white")

        # random expectation. Mean at 0 by definition.
        # Use the bias_std to sample from
        # do one instance only
        random_means = []
        for session in sessions_used[i]:
            # get x number of a random bias
            sess_std = random_opto_df[
                random_opto_df.SessionID == session
            ].bias_std.values
            random_means.append(
                np.random.normal(loc=0.0, scale=sess_std[0], size=100)
            )
        random_means_flat_list = [
            item for sublist in random_means for item in sublist
        ]

        spr_adj = 1.5
        bp = ax.boxplot(
            random_means_flat_list,
            positions=[spread * spr_adj],
            widths=0.07,
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
            plt.setp(bp[element], color="lightgray", linewidth=3)
        for patch in bp["boxes"]:
            patch.set(facecolor="white")

        pvals.append(
            stats.kruskal(mean_vals[i], random_means_flat_list).pvalue
        )

    # add pvalues info
    hlocs = [20, -20]
    hadj = [1.2, 1.4]

    for i, ax in enumerate(axs):
        pvaltext = "{0:.7f}".format(pvals[i])
        ax.text(
            x=spread * (1 + spr_adj) / 2,
            y=hlocs[i] * hadj[i],
            s="pval {}".format(str(pvaltext)),
            horizontalalignment="center",
            fontsize=14,
        )
        ax.plot(
            [spread, spread * spr_adj],
            [hlocs[i], hlocs[i]],
            color="k",
            linewidth=0.5,
        )
        ax.plot(
            [spread, spread],
            [hlocs[i], hlocs[i] * 0.8],
            color="k",
            linewidth=0.5,
        )
        ax.plot(
            [spread * spr_adj, spread * spr_adj],
            [hlocs[i], hlocs[i] * 0.8],
            color="k",
            linewidth=0.5,
        )
        ax.set_xticks([])

    return fig


def make_figure_optoinhibition_after_learning_curves(oal_df, random_opto_df):
    # Plot the data with the error bars for the random sampling,
    # and the custom fitting
    ColorList = ["powderblue", "plum"]
    normal_color = "gray"
    LabelList = ["left stimulation", "right stimulation"]
    Genotypes = ["D1opto", "D2opto"]
    StimSides = ["Left", "Right"]

    n_cols = 2

    fig, axs = plt.subplots(
        1, n_cols, figsize=(7 * n_cols, 5), facecolor="w", edgecolor="k"
    )

    axs = axs.ravel()

    for i, ax in enumerate(axs):

        genot = Genotypes[i]

        # select sessions
        g_mask = random_opto_df.Genotype == genot
        s_mask = random_opto_df.stimulated_side.isin(StimSides)

        sessions_list_cleaned = random_opto_df[
            np.logical_and(g_mask, s_mask)
        ].SessionID

        # plot the normal choices and fit
        session_df = oal_df[oal_df["SessionID"].isin(sessions_list_cleaned)]
        df_for_plot = session_df[session_df.OptoStim == 0]
        plot_utils.plot_regression(
            df=df_for_plot,
            ax=ax,
            color=normal_color,
            label="",
            plot_points=False,
        )

        predictDif, PsyPer, _, _, EB = cuf.PP_ProcessExperiment(
            df_for_plot, 0, error_bars="SessionTime"
        )
        plot_utils.PlotPsychPerformance(
            dataDif=PsyPer["Difficulty"],
            dataPerf=PsyPer["Performance"],
            predictDif=predictDif,
            ax=ax,
            fakePred=None,
            realPred=None,
            color=normal_color,
            label="control trials",
            errorBars=EB,
        )

        # plot each side
        for k, stside in enumerate(StimSides):
            s_mask = random_opto_df.stimulated_side == stside
            sessions_list_cleaned = random_opto_df[
                np.logical_and(g_mask, s_mask)
            ].SessionID

            # plot the normal choices and fit
            session_df = oal_df[
                oal_df["SessionID"].isin(sessions_list_cleaned)
            ]
            df_for_plot = session_df[session_df.OptoStim == 1]
            plot_utils.plot_regression(
                df=df_for_plot,
                ax=ax,
                color=ColorList[k],
                label="",
                plot_points=False,
            )

            predictDif, PsyPer, _, _, EB = cuf.PP_ProcessExperiment(
                df_for_plot, 0, error_bars="SessionTime"
            )
            plot_utils.PlotPsychPerformance(
                dataDif=PsyPer["Difficulty"],
                dataPerf=PsyPer["Performance"],
                predictDif=predictDif,
                ax=ax,
                fakePred=None,
                realPred=None,
                color=ColorList[k],
                label=LabelList[k],
                errorBars=EB,
            )

        ax.text(
            0.5,
            1.05,
            genot,
            horizontalalignment="center",
            fontweight="bold",
            transform=ax.transAxes,
            fontsize=16,
        )

        ax.axis("on")
        # remove some ticks
        ax.tick_params(
            which="both",
            top=False,
            bottom="on",
            left="on",
            right=False,
            labelleft="on",
            labelbottom="on",
        )
        try:
            if not ax.is_first_col():
                ax.set_ylabel("")
                ax.set_yticks([])
            if not ax.is_last_row():
                ax.set_xlabel("")
                ax.set_xticks([])
        except AttributeError:
            pass

        ax.set_ylim(-2.0, 102.0)
        ax.legend(bbox_to_anchor=(1.05, 1), loc=0, borderaxespad=0.0)

        # get rid of the frame
        for spine in ax.spines.values():
            spine.set_visible(True)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

        # tick text size
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(14)

        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(14)

        # reverse x axis ticks
        ax.set_xticklabels([2, 18, 34, 50, 66, 82, 98][::-1])

        ax.set_ylabel("trials reported low (%)", fontsize=16)

        ax.set_xlabel("low tones (%)", fontsize=16)

    plt.tight_layout()

    return fig


def make_figure_6ohda_lesion_correlation(merged_df, color_palette):
    fig = plt.figure(figsize=(5, 3))
    sns.scatterplot(
        data=merged_df[merged_df.ExperimentalGroup == "6OHDAtail"],
        x="maximum_performance",
        y="ratio posterior/anterior",
        color=color_palette[1],
        s=100,
    )
    ax = plt.gca()
    cntrls = merged_df[merged_df.ExperimentalGroup == "CortexBuffer"].copy()
    ax.scatter(
        cntrls.maximum_performance.mean(),
        cntrls["ratio posterior/anterior"].mean(),
        marker="o",
        linewidth=3,
        s=100,
        facecolors="none",
        edgecolors=color_palette[0],
    )

    # calculate regression with the 6OHDAs
    df_for_reg = (
        merged_df[merged_df.ExperimentalGroup == "6OHDAtail"].dropna().copy()
    )
    plot_utils.reg_in_ax(
        df_for_reg["maximum_performance"],
        df_for_reg["ratio posterior/anterior"],
        ax,
        "upper left",
    )
    # ax.get_legend().remove()
    ax.set_ylabel("posterior DA / anterior DA")
    ax.set_xlabel("maximum performance (%)")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    return fig


def make_figure_optoinhibition_through_learning_per_mouse(
    random_opto_df, significance=0.05
):

    stim_types = ["Right", "Left"]
    colors = ["c", "m"]
    animals = pd.unique(random_opto_df.AnimalID)
    fig, axs = plt.subplots(
        len(pd.unique(random_opto_df.Genotype)),
        5,
        figsize=(15, len(animals)),
        sharex=True,
        sharey=True,
    )
    axs = axs.ravel()
    # for ax in axs.ravel(): ax.axis('off')
    for an_c, animal in enumerate(animals):
        ax = axs[an_c]
        ax.set_title(animal)
        ax.axhline(0, color="grey", linestyle="--")
        animal_rdf = random_opto_df[random_opto_df.AnimalID == animal]

        for session in pd.unique(animal_rdf.SessionID):
            session_idx = animal_rdf.index[
                animal_rdf.SessionID == session
            ].item()
            st_t = animal_rdf.loc[session_idx].stimulated_side
            sp = animal_rdf.loc[session_idx].session_performance
            cb = animal_rdf.loc[session_idx].contralateral_bias_exp
            cbs = animal_rdf.loc[session_idx].bias_std_exp * 2
            st_idx = stim_types.index(st_t)

            ax.plot(
                [sp, sp],
                [cb - cbs, cb + cbs],
                color=colors[st_idx],
                linewidth=3,
                alpha=0.5,
            )
            facecol = "w"
            edgecol = colors[st_idx]
            if animal_rdf.loc[session_idx].significance_value < significance:
                facecol = colors[st_idx]
                edgecol = "k"
            ax.plot(
                sp,
                cb,
                "o",
                ms=14,
                markerfacecolor=facecol,
                alpha=1,
                color=edgecol,
            )
        ax.set_xlabel("Performance")
        ax.set_ylabel("Contralateral bias")

    return fig


def make_figure_optoinhibition_through_learning(
    random_opto_df,
    reg_dicc,
    xs,
    group_shuffle_mean,
    group_shuffle_std,
    significance=0.05,
    colors=["b", "m"],
):
    # plot them all together
    genotypes = ["D1opto", "D2opto"]
    labels_for_legend = ["D1-Arch", "D2-Arch"]

    fig, ax = plt.subplots(1, 1, figsize=(20, 10))
    ax.axhline(0, color="grey", linestyle="--", linewidth=1)

    fibers = pd.unique(random_opto_df.fiber_id)

    for an_c, fiber in enumerate(fibers):

        animal_rdf = random_opto_df[random_opto_df.fiber_id == fiber]

        for session in pd.unique(animal_rdf.SessionID):
            session_idx = animal_rdf.index[
                animal_rdf.SessionID == session
            ].item()
            sp = animal_rdf.loc[session_idx].session_performance
            cb = animal_rdf.loc[session_idx].contralateral_bias_exp
            cbs = animal_rdf.loc[session_idx].bias_std_exp * 2
            genot = animal_rdf.loc[session_idx].Genotype
            g_idx = genotypes.index(genot)
            gfl = labels_for_legend[g_idx]

            ax.plot(
                [sp, sp],
                [cb - cbs, cb + cbs],
                color=colors[g_idx],
                linewidth=4,
                alpha=0.8,
            )
            facecol = "w"
            edgecol = colors[g_idx]
            if animal_rdf.loc[session_idx].significance_value < significance:
                facecol = colors[g_idx]
                edgecol = "k"
            ax.plot(
                sp,
                cb,
                "o",
                ms=20,
                markerfacecolor=facecol,
                alpha=1,
                color=edgecol,
                label=gfl,
            )

    # add regression
    for genotype in np.unique(reg_dicc["genotypes"]):
        # find indexes for that genotype
        gmask = np.where([g == genotype for g in reg_dicc["genotypes"]])[0]

        # get means and stds
        fits_array = np.zeros([len(gmask), len(xs)])
        for i, idx in enumerate(gmask):
            fits_array[i, :] = reg_dicc["fits"][idx]
        genot_means = np.mean(fits_array, axis=0)
        genot_stds = np.std(fits_array, axis=0)

        genot_idx = genotypes.index(genotype)

        plt.plot(
            xs, genot_means, color=colors[genot_idx], zorder=-30
        )  # , label=labels_for_legend[genot_idx])
        plt.fill_between(
            xs,
            genot_means - genot_stds,
            genot_means + genot_stds,
            color=colors[genot_idx],
            alpha=0.2,
            zorder=-50,
        )

    # plot random
    plt.fill_between(
        xs,
        group_shuffle_mean - group_shuffle_std,
        group_shuffle_mean + group_shuffle_std,
        color="grey",
        alpha=0.2,
        zorder=-100,
    )

    # beautify the plot
    ax.set_ylim(-75, 75)
    ax.set_xlim(50, 100)

    # get rid of the frame
    for spine in ax.spines.values():
        spine.set_visible(False)

    # set labels
    ax.set_ylabel("Contralateral bias (% of choices)", fontsize=20)
    ax.set_xlabel("Task performance (% of correct choices)", fontsize=20)
    ax.set_title(
        "Optoinhibition effects increase through learning", fontsize=25
    )

    # add legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(
        by_label.values(),
        by_label.keys(),
        frameon=False,
        loc=(0.32, 0.85),
        ncol=2,
        fontsize=20,
    )
    # ax.legend()
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(18)

    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(18)

    return fig


def make_figure_optoinhibition_significant_sessions(sess_df):
    fig, ax = plt.subplots(1, 1, figsize=(20, 5))
    sns.lineplot(
        data=sess_df,
        x="performance_window",
        y="proportion_of_significant_sessions",
        ci=None,
        ax=ax,
        linewidth=3,
    )
    ax.lines[0].set_linestyle("--")

    # add number of sessions tested
    for idx in sess_df.index:
        ax.text(
            sess_df.loc[idx, "performance_window"],
            sess_df.loc[idx, "proportion_of_significant_sessions"] + 5,
            "{}/{}".format(
                sess_df.loc[idx, "number_of_significant_sessions"],
                sess_df.loc[idx, "n_sessions"],
            ),
            horizontalalignment="center",
            color="black",
            weight="semibold",
            fontsize=16,
        )
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    # set labels
    ax.set_ylabel("Percentage of biased sessions", fontsize=20)
    ax.set_xlabel("Task performance (% of correct choices)", fontsize=20)
    # ax.set_title('Optoinhibition effects increase
    # through learning', fontsize=25)

    # ax.legend()
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(18)

    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(18)

    ax.set_xlim(50, 100)

    return fig


# see all the sessions done per animal
def make_figure_opto_da_all_mice(
    dao_df, ini_trials, ao_trials, example_session
):

    # Plot the data with the error bars for the random sampling,
    # and the custom fitting
    BRS = ["tStr", "NAc"]
    PS = ["Left", "Right"]
    PI = ["Center", "Side"]

    CondList = [
        (dao_df["TrialIndex"] < ini_trials),
        (dao_df["TrialIndex"] > ao_trials),
    ]
    ColorList = ["black", "red"]
    LabelList = ["Normal", "After opto"]

    n_cols = dao_df.groupby("AnimalID")["SessionID"].nunique().max()

    fig, axs = plt.subplots(
        len(pd.unique(dao_df["AnimalID"])),
        n_cols,
        figsize=(5 * n_cols, 5 * len(pd.unique(dao_df["AnimalID"]))),
        facecolor="w",
        edgecolor="k",
    )

    fig.subplots_adjust(hspace=0.2, wspace=1)

    for ax in axs.ravel():
        ax.axis("off")

    for an_counter, animal in enumerate(pd.unique(dao_df["AnimalID"])):
        animal_df = dao_df[dao_df["AnimalID"] == animal]

        for counter, session in enumerate(pd.unique(animal_df["SessionID"])):
            session_df = dao_df[dao_df["SessionID"] == session]

            ax = axs[an_counter, counter]
            ax.hlines(50, 0, 100, linestyles="dotted", alpha=0.4)

            # overlay fits
            for i, condition in enumerate(CondList):
                plot_utils.plot_regression(
                    df=session_df[condition],
                    ax=ax,
                    color=ColorList[i],
                    label=LabelList[i],
                    plot_points=True,
                )
            ax.set_ylabel("Percentage of rightward choices", fontsize=16)
            ax.set_xlabel("Evidence of high frequency", fontsize=16)

            ax.axis("on")
            # remove some ticks
            ax.tick_params(
                which="both",
                top=False,
                bottom="on",
                left="on",
                right=False,
                labelleft="on",
                labelbottom="on",
            )
            try:
                if not ax.is_first_col():
                    ax.set_ylabel("")
                    ax.set_yticks([])
                if not ax.is_last_row():
                    ax.set_xlabel("")
                    ax.set_xticks([])
            except AttributeError:
                pass

            ax.set_ylim(-2.0, 102.0)
            # ax.legend(bbox_to_anchor=(1.05, 1), loc=0, borderaxespad=0.)
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

            plt.tight_layout()

            ax.invert_xaxis()

            ax.get_legend().remove()
            ax.text(
                0.5,
                1.05,
                str(counter) + ": " + (session),
                horizontalalignment="center",
                fontweight="bold",
                transform=ax.transAxes,
            )
            ax.text(
                0.5,
                1,
                "No of trials: "
                + str(len(animal_df[(animal_df["SessionID"] == session)])),
                horizontalalignment="center",
                transform=ax.transAxes,
            )
            # Fiber location
            fl = BRS[int(session_df.iloc[0].FullGUI["FiberLocation"]) - 1]
            # Side
            side = session_df.Stimulation.unique()[0]
            ax.text(
                0.5,
                0.95,
                "Stim location: " + fl + " " + side,
                horizontalalignment="center",
                transform=ax.transAxes,
            )
            # Port stimulated
            port = PI[int(session_df.iloc[0].FullGUI["OptoState"]) - 1]
            portside = PS[int(session_df.iloc[0].FullGUI["JOPSide"]) - 1]
            ax.text(
                0.5,
                0.9,
                portside + " stimulation in " + port + " port",
                horizontalalignment="center",
                transform=ax.transAxes,
            )

            if session == example_session:
                ax.set_facecolor("lightgrey")

        update_progress((an_counter + 1) / len(pd.unique(dao_df["AnimalID"])))

    return fig


def make_figure_opto_da_boxplots(opto_df_sel, hor):
    # make a palette
    colors = ["darkslategray", "sandybrown"]

    # spread of things
    spread = 0.2
    randspread = 2.3 * spread

    # plot
    rel = sns.catplot(
        data=opto_df_sel,
        x="FiberArea",
        y="BiasToStimPort",
        hue="FiberArea",
        alpha=1,
        size=5,
        marker="o",
        s=10,
        linewidth=1,
        edgecolor="k",  # jitter=.15,
        hue_order=hor,
        kind="swarm",
        height=15,
        aspect=8 / 6,
        palette=sns.color_palette(colors),
    )

    # add the distribution of the random biases, and mean and std
    axs = rel.fig.axes

    orig_y_lim = axs[0].get_ylim()[1]

    # move overall title up
    rel.fig.subplots_adjust(top=0.9)
    rel.fig.suptitle(
        "Contralateral DA stimulation on center port", y=1.05, fontsize=16
    )
    for ax in axs:
        ax.set_ylabel("Bias to stimulated port (% of choices)", fontsize=16)
        ax.set_xlabel("")
        ax.hlines(
            0,
            ax.get_xlim()[0],
            ax.get_xlim()[1],
            linestyles="dotted",
            alpha=0.4,
            zorder=-2,
        )
        for i, pos in enumerate(ax.get_xticks()):
            facond = opto_df_sel.FiberArea == hor[i]
            randbiases = np.array(
                [
                    item
                    for sublist in opto_df_sel[facond].RandomBiases.values
                    for item in sublist
                ]
            )
            bp = ax.boxplot(
                randbiases,
                positions=[pos + randspread],
                widths=0.07,
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
                plt.setp(bp[element], color="gray", linewidth=3)
            for patch in bp["boxes"]:
                patch.set(facecolor="white")

            # mean and std
            dist_to_compare = opto_df_sel[facond].BiasToStimPort.values
            bp = ax.boxplot(
                dist_to_compare,
                positions=[pos + spread],
                widths=0.07,
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
                plt.setp(bp[element], color=colors[i], linewidth=3)
            for patch in bp["boxes"]:
                patch.set(facecolor="white")

            pval = stats.ranksums(
                dist_to_compare, randbiases, alternative="greater"
            ).pvalue
            ax.text(
                x=i + 1.5 * spread,
                y=1.2 * orig_y_lim,
                s="pval = {:.6}".format(str(pval)),
                horizontalalignment="center",
                fontsize=14,
            )
            ax.plot(
                [i + spread * 0.5, i + randspread],
                [orig_y_lim * 1.1, orig_y_lim * 1.1],
                color="k",
                linewidth=0.5,
            )
            ax.plot(
                [i + spread * 0.5, i + 0.5 * spread],
                [orig_y_lim, orig_y_lim * 1.1],
                color="k",
                linewidth=0.5,
            )
            ax.plot(
                [i + randspread, i + randspread],
                [orig_y_lim, orig_y_lim * 1.1],
                color="k",
                linewidth=0.5,
            )

        ax.set_xlim([-2 * spread, 1 + 5 * spread])
        ax.set_xticks([spread, 1 + spread])

        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(16)
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(14)

        # keep only y axis and floating x ticks
        ax.set_frame_on(False)
        xmin, xmax = ax.get_xaxis().get_view_interval()
        ax.plot((xmin, xmin), (-30, 30), color="black", linewidth=1)

        ax.set_xticklabels(hor)

    return rel

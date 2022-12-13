import numpy as np
import pandas as pd
import scipy.optimize as opt
from IPython.display import clear_output

from mouse_behavior_analysis_tools.utils import model_utils
from mouse_behavior_analysis_tools.utils.misc_utils import update_progress


def get_df_of_behavior_model_by_animal(df_to_plot, mouse_max_perf):

    ans_list = np.sort(df_to_plot.AnimalID.unique())
    num_ans = len(ans_list)
    # create a diccionary to store the results, and lists to rescale the data
    fit_dir = {}
    xmeans_list = []
    xsd_list = []

    # process data from all animals
    for counter, animal in enumerate(ans_list):
        df = df_to_plot[df_to_plot.AnimalID == animal][
            [
                "CumulativeTrialNumberByProtocol",
                "CurrentPastPerformance100",
                "SessionID",
            ]
        ].dropna()

        # define values
        xdata = np.array(df.CumulativeTrialNumberByProtocol)
        ydata = np.array(df.CurrentPastPerformance100)

        # scale the data
        xdatasc = (xdata - xdata.mean()) / xdata.std()
        ydatasc = ydata / 100

        # limit to the maximum performance for this mouse:
        mp = (
            mouse_max_perf[
                mouse_max_perf.AnimalID == animal
            ].CurrentPastPerformance100.iloc[0]
            / 100
        )

        cost_func = lambda x: np.mean(
            np.abs(
                model_utils.sigmoid_func(xdatasc, x[0], x[1], x[2]) - ydatasc
            )
        )
        res = opt.minimize(
            cost_func, [1, 0, 0], bounds=((0.5, mp), (0.0, 10.0), (None, None))
        )

        update_progress(counter / num_ans)

        # update dicctionary and lists
        fit_dir[animal] = res
        xmeans_list.append(xdata.mean())
        xsd_list.append(xdata.std())

    # convert to dataframe
    fit_df = pd.DataFrame(
        {
            "AnimalID": list(fit_dir.keys()),
            "maximum_performance": [v.x[0] for k, v in fit_dir.items()],
            "slope": [v.x[1] for k, v in fit_dir.items()],
            "bias": [v.x[2] for k, v in fit_dir.items()],
        }
    )
    # get the Experimental procedure
    fit_df["ExperimentalGroup"] = fit_df["AnimalID"].apply(
        lambda x: df_to_plot[
            df_to_plot.AnimalID == x
        ].ExperimentalGroup.unique()[0]
    )
    # rescale back the coefficients
    fit_df.maximum_performance = fit_df.maximum_performance * 100
    fit_df.slope = fit_df.slope / np.array(xsd_list)
    fit_df.bias = fit_df.bias * np.array(xsd_list) + np.array(xmeans_list)

    update_progress(1)
    return fit_df


def get_steepest_point_of_slope(fit_df):
    der_max_dir = {}
    ans_list = np.sort(fit_df.AnimalID.unique())
    for animal in ans_list:
        m_point = opt.fmin(
            lambda x: -model_utils.der_sig(
                x,
                *[
                    fit_df[fit_df.AnimalID == animal].maximum_performance.iloc[
                        0
                    ],
                    fit_df[fit_df.AnimalID == animal].slope.iloc[0],
                    fit_df[fit_df.AnimalID == animal].bias.iloc[0],
                ],
            ),
            0,
            full_output=True,
        )

        der_max_dir[animal] = (m_point[0][0], -m_point[1])

    clear_output()

    update_progress(1)
    return der_max_dir

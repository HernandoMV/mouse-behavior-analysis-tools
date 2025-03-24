import numpy as np
import pandas as pd
import scipy.optimize as opt
from IPython.display import clear_output

from mouse_behavior_analysis_tools.utils import model_utils
from mouse_behavior_analysis_tools.utils.misc_utils import update_progress


def get_df_of_behavior_model_by_animal_DEPRECATED(df_to_plot, mouse_max_perf):

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


def get_df_of_behavior_model_by_animal(df_to_plot, mouse_max_perf):

    ans_list = np.sort(df_to_plot.AnimalID.unique())
    num_ans = len(ans_list)
    # create a diccionary to store the results, and lists to rescale the data
    fit_dir = {}
    xmeans_list = []
    xsd_list = []
    der_max_list = []
    sum_of_squared_errors_weibull = []
    sum_of_squared_errors_sigmoid = []
    sig1_list = []
    sig2_list = []
    sig3_list = []

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

        # Define a cost function: here using the mean squared error
        def cost_func(x):
            return np.mean(
                np.square(
                    model_utils.weibull_func(xdata, x[0], x[1], x[2]) - ydata
                )
            )

        # Set up a better initial guess based on expected range
        initial_guess = [50, 700, .5]  # Since x and y are scaled, start with values in normalized range
        bounds = ((0.1, 100), (0, 5000), (0.1, 5.0))  # Adjusted bounds to better reflect expected parameter range

        # Step 2: Perform the optimization on the normalized data using a different method ('Powell' or 'L-BFGS-B')
        res = opt.minimize(cost_func, initial_guess, method='Powell', bounds=bounds)

        # Calculate the maximum of the derivative function
        xs = np.arange(0, 5000, 1)
        der_max = np.max(model_utils.derivative_weibull_func(xs, *res.x))
        der_max_list.append(der_max)
        weibull_values = model_utils.weibull_func(xdata, *res.x)
        sum_of_squared_errors_weibull.append(np.sum((weibull_values - ydata) ** 2))

        # Calculate the sigmoid fitting error

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
        cost_func_s = lambda x: np.mean(
            np.abs(
                model_utils.sigmoid_func(xdatasc, x[0], x[1], x[2]) - ydatasc
            )
        )
        res_sig = opt.minimize(
            cost_func_s, [1, 0, 0], bounds=((0.5, mp), (0.0, 10.0), (None, None))
        )
        # scale the values back
        sig1 = res_sig.x[0] * 100
        sig2 = res_sig.x[1] / xdata.std()
        sig3 = res_sig.x[2] * xdata.std() + xdata.mean()

        sigmoid_values = model_utils.sigmoid_func_sc(xdata, sig1, sig2, sig3)
        sum_of_squared_errors_sigmoid.append(np.sum((sigmoid_values - ydata) ** 2))

        update_progress(counter / num_ans)

        # update dicctionary and lists
        fit_dir[animal] = res
        xmeans_list.append(xdata.mean())
        xsd_list.append(xdata.std())
        sig1_list.append(sig1)
        sig2_list.append(sig2)
        sig3_list.append(sig3)

    # convert to dataframe
    fit_df = pd.DataFrame(
        {
            "AnimalID": list(fit_dir.keys()),
            "a": [v.x[0] for k, v in fit_dir.items()],
            "l": [v.x[1] for k, v in fit_dir.items()],
            "s": [v.x[2] for k, v in fit_dir.items()],
            "max_of_der": der_max_list,
            "sum_of_squared_errors_weibull": sum_of_squared_errors_weibull,
            "sum_of_squared_errors_sigmoid": sum_of_squared_errors_sigmoid,
            "sig1": sig1_list,
            "sig2": sig2_list,
            "sig3": sig3_list,
        }
    )
    # get the Experimental procedure
    fit_df["ExperimentalGroup"] = fit_df["AnimalID"].apply(
        lambda x: df_to_plot[
            df_to_plot.AnimalID == x
        ].ExperimentalGroup.unique()[0]
    )
    # rescale back the coefficients
    # fit_df.a = fit_df.a * 100
    # fit_df.s = fit_df.s / np.array(xsd_list)
    # fit_df.l = fit_df.l * np.array(xsd_list) + np.array(xmeans_list)

    update_progress(1)
    return fit_df
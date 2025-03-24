# define the models for the fitting
import numpy as np


# sigmoid function
def sigmoid_func(x, perf_end, slope, bias):
    return (perf_end - 0.5) / (1 + np.exp(-slope * (x - bias))) + 0.5


# sigmoid function scaled
def sigmoid_func_sc(x, perf_end, slope, bias):
    return (perf_end - 50) / (1 + np.exp(-slope * (x - bias))) + 50


# derivative function
def der_sig(x, perf_end, slope, bias):
    return (
        (perf_end - 50)
        * slope
        * np.exp(-slope * (x - bias))
        / (1 + np.exp(-slope * (x - bias))) ** 2
    )


# define this weibull y = A {1 – 2^ – [(x/L)^S]}
def weibull_func(x, a, l, s):
    return 50 + a * (
        1 - 2 **(
            -(x / l) ** s)
    )


def derivative_weibull_func(x, a, l, s):
    ln2 = np.log(2)
    term1 = a * ln2 * s
    term2 = (x ** (s - 1)) / (l ** s)
    term3 = 2 ** (-(x / l) ** s)

    return term1 * term2 * term3
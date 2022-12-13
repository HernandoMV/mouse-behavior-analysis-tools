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

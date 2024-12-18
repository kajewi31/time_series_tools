import numpy as np


def fip_extrema(arr, window):
    from scipy.signal import argrelextrema

    local_maxima = argrelextrema(arr, np.greater, order=window)[0]
    local_minima = argrelextrema(arr, np.less, order=window)[0]

    inflection_points = sorted(np.concatenate((local_maxima, local_minima)))
    return inflection_points

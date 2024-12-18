import numpy as np
import pandas as pd


def moving_average_smooth(arr, window):
    """
    Apply Moving Average Smoothing to an array.

    Parameters:
        arr (array-like): The input array to smooth.
        window (int): The size of the moving average window.
            - 1 <= window

    Returns:
        np.ndarray: The smoothed array with NaN padding at the beginning.
    """
    smoothed_arr = pd.Series(arr).rolling(window=window).mean().to_numpy()
    return smoothed_arr


def exponential_moving_average_smooth(arr, alpha):
    """
    Apply Exponential Moving Average (EMA) Smoothing to an array.

    Parameters:
        arr (array-like): The input array to smooth.
        alpha (float): Smoothing factor. Smaller values give more smoothing.
            - 0 < alpha <= 1

    Returns:
        np.ndarray: The smoothed array.
    """
    smoothed_arr = pd.Series(arr).ewm(alpha=alpha, adjust=False).mean().to_numpy()
    return smoothed_arr


def savgol_filter_smooth(arr, window, polyorder):
    """
    Apply Savitzky-Golay Filter Smoothing to an array.

    Parameters:
        arr (array-like): The input array to smooth.
        window (int): The length of the filter window (odd integer).
            - 3 <= window
        polyorder (int): The polynomial order to fit the window.
            - 0 <= polyorder < window

    Returns:
        np.ndarray: The smoothed array.
    """
    from scipy.signal import savgol_filter

    smoothed_arr = savgol_filter(arr, window_length=window, polyorder=polyorder)
    return smoothed_arr


def lowess_smooth(arr, frac):
    """
    Apply LOWESS (Locally Weighted Scatterplot Smoothing) to an array.

    Parameters:
        arr (array-like): The input array to smooth.
        frac (float): The fraction of data used to compute the smoothing at each point.
            - 0 < frac <= 1

    Returns:
        np.ndarray: The smoothed array.
    """
    from statsmodels.nonparametric.smoothers_lowess import lowess

    x = np.arange(1, len(arr) + 1)
    y = arr
    result = lowess(y, x, frac=frac)
    smoothed_arr = result[:, 1]
    return smoothed_arr


def spline_smooth(arr, s):
    """
    Apply Spline Smoothing to an array using UnivariateSpline.

    Parameters:
        arr (array-like): The input array to smooth.
        s (float): Smoothing factor. Larger values result in more smoothing.
            - 0 <= s

    Returns:
        np.ndarray: The smoothed array.
    """
    from sklearn.preprocessing import MinMaxScaler
    from scipy.interpolate import UnivariateSpline

    x = np.arange(1, len(arr) + 1)
    y = arr.reshape(-1, 1)

    mms_y = MinMaxScaler()
    scaled_y = mms_y.fit_transform(y).flatten()

    spline = UnivariateSpline(x, scaled_y, s=s)
    scaled_smoothed_y = spline(x)

    smoothed_arr = mms_y.inverse_transform(scaled_smoothed_y.reshape(-1, 1)).flatten()
    return smoothed_arr


def gaussian_smooth(arr, sigma):
    """
    Apply Gaussian Smoothing to an array.

    Parameters:
        arr (array-like): The input array to smooth.
        sigma (float): The standard deviation of the Gaussian kernel.
            - 0 < sigma

    Returns:
        np.ndarray: The smoothed array with NaN padding at the boundaries.
    """
    from scipy.signal import convolve

    size = int(3 * abs(sigma) + 1)
    x = np.arange(-size, size + 1)
    kernel = np.exp(-(x**2) / (2 * sigma**2))
    kernel = kernel / np.sum(kernel)

    smoothed_arr = convolve(arr, kernel, mode='valid')
    padding_size = len(arr) - len(smoothed_arr)
    left_pad = padding_size // 2
    right_pad = padding_size - left_pad

    smoothed_arr = np.pad(smoothed_arr, (left_pad, right_pad), constant_values=np.nan)
    return smoothed_arr

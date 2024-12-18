import numpy as np
import pandas as pd


def moving_average_smooth(series, window):
    """
    Apply Moving Average Smoothing to a Pandas Series.

    Parameters:
        series (pd.Series): The input Pandas Series to smooth.
        window (int): The size of the moving average window.
            - 1 <= window

    Returns:
        pd.Series: The smoothed series.
    """
    smoothed_series = series.rolling(window=window).mean()
    return smoothed_series


def exponential_moving_average_smooth(series, alpha):
    """
    Apply Exponential Moving Average (EMA) Smoothing to a Pandas Series.

    Parameters:
        series (pd.Series): The input Pandas Series to smooth.
        alpha (float): The smoothing factor. Smaller values give more smoothing.
            - 0 < alpha <= 1

    Returns:
        pd.Series: The smoothed series.
    """
    smoothed_series = series.ewm(alpha=alpha, adjust=False).mean()
    return smoothed_series


def savgol_filter_smooth(series, window, polyorder):
    """
    Apply Savitzky-Golay Filter Smoothing to a Pandas Series.

    Parameters:
        series (pd.Series): The input Pandas Series to smooth.
        window (int): The length of the filter window (must be odd).
            - 3 <= window
        polyorder (int): The polynomial order to fit the window.
            - 0 <= polyorder < window

    Returns:
        pd.Series: The smoothed series.
    """
    from scipy.signal import savgol_filter
    smoothed_arr = savgol_filter(series, window_length=window, polyorder=polyorder)
    smoothed_series = pd.Series(smoothed_arr, index=series.index, name=series.name)
    return smoothed_series


def lowess_smooth(series, frac):
    """
    Apply LOWESS (Locally Weighted Scatterplot Smoothing) to a Pandas Series.

    Parameters:
        series (pd.Series): The input Pandas Series to smooth.
        frac (float): The fraction of data used to compute the smoothing at each point (between 0 and 1).
            - 0 < frac <= 1

    Returns:
        pd.Series: The smoothed series.
    """
    from statsmodels.nonparametric.smoothers_lowess import lowess

    x = np.arange(1, len(series) + 1)
    y = series.to_numpy()
    result = lowess(y, x, frac=frac)
    smoothed_arr = result[:, 1]
    smoothed_series = pd.Series(smoothed_arr, index=series.index, name=series.name)
    return smoothed_series


def spline_smooth(series, s):
    """
    Apply Spline Smoothing to a Pandas Series using UnivariateSpline.

    Parameters:
        series (pd.Series): The input Pandas Series to smooth.
        s (float): The smoothing factor. Larger values result in more smoothing.
            - 0 <= s

    Returns:
        pd.Series: The smoothed series.
    """
    from sklearn.preprocessing import MinMaxScaler
    from scipy.interpolate import UnivariateSpline

    x = np.arange(1, len(series) + 1)
    y = series.to_numpy().reshape(-1, 1)

    mms_y = MinMaxScaler()
    scaled_y = mms_y.fit_transform(y).flatten()

    spline = UnivariateSpline(x, scaled_y, s=s)
    scaled_smoothed_y = spline(x)
    smoothed_arr = mms_y.inverse_transform(scaled_smoothed_y.reshape(-1, 1)).flatten()
    smoothed_series = pd.Series(smoothed_arr, index=series.index, name=series.name)

    return smoothed_series


def gaussian_smooth(series, sigma):
    """
    Apply Gaussian Smoothing to a Pandas Series.

    Parameters:
        series (pd.Series): The input Pandas Series to smooth.
        sigma (float): The standard deviation of the Gaussian kernel.
            - 0 < sigma

    Returns:
        pd.Series: The smoothed series.
    """
    from scipy.signal import convolve

    size = int(3 * abs(sigma) + 1)
    x = np.arange(-size, size + 1)
    kernel = np.exp(-(x**2) / (2 * sigma**2))
    kernel = kernel / np.sum(kernel)

    smoothed_arr = convolve(series.to_numpy(), kernel, mode='valid')
    padding_size = len(series) - len(smoothed_arr)
    left_pad = padding_size // 2
    right_pad = padding_size - left_pad

    padded_arr = np.pad(smoothed_arr, (left_pad, right_pad), constant_values=np.nan)

    smoothed_series = pd.Series(padded_arr, index=series.index, name=series.name)
    return smoothed_series
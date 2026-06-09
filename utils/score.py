import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import utils.data as datutils
from multicam.qt import qt_gauss, qt_gauss_base, qt_inverse_gauss_base    # type: ignore


def weight_split(weights, x):
    abs_weights = np.abs(weights)
    sq_weights = weights**2
    weights_sums = np.sum(weights, axis=1)
    abs_sums = np.sum(abs_weights, axis=1)
    sq_sums = np.sum(sq_weights, axis=1)
    # where i expect to see the most impact from features
    indx = np.argmin(weights_sums)
    aindx = np.argmax(abs_sums)
    sqindx = np.argmax(sq_sums)

    wsum_ds = x.dot(weights.T)    # (n_regions, scale)
    abs_wsum_ds = x.dot(abs_weights.T)
    sq_wsum_ds = x.dot(sq_weights.T)
    inds = (indx, aindx, sqindx)
    return wsum_ds, abs_wsum_ds, sq_wsum_ds, inds


def build_rho(df: pd.DataFrame, mah) -> np.ndarray:
    """Build spearman correlation matrix between parameters and mass accretion history, at each time step

    Parameters
    ----------
    df : pd.DataFrame
       ds or sm parameters
    mah : np.ndarray
        mass accretion history array, ma or am

    Returns
    -------
    np.ndarray
        Spearman correlation matrix
    """
    n_times = np.shape(mah)[1]
    n_features = np.shape(df)[1]
    rho = np.zeros((n_times, n_features))
    r = 1000
    df_r = df.reset_index(drop=True)
    for i in range(n_times):
        mm0 = pd.Series(mah[:, i])
        df0 = pd.concat([mm0, df_r], axis=1)
        for _ in range(r):
            bootdf = df0.sample(n=len(df0), replace=True)
            sp, _ = spearmanr(bootdf)
            w = sp[1:, 0]/r
            rho[i, :] += w
    return rho


def wsum(xng, weights: np.ndarray, model=None) -> np.ndarray:
    """Weighted sum of cluster parameters.

    Parameters
    ----------
    xng : np.ndarray or pd.DataFrame
        ds or sm parameters, not yet quantile transformed
    weights : np.ndarray
        multicam training weights or spearman correlation coefficients
    model : MultiCAM model
        model fitted on training data, used to gaussianize the features + predicted ma and invert it back to data space

    Returns
    -------
    np.ndarray
        scores for each halo as a function of time
    """
    if isinstance(xng, pd.DataFrame):
        xng = xng.to_numpy()

    if model is None:
        xg = qt_gauss(xng, axis=0)  # gaussianize x
        # normalize weights by sum of absolute values
        weights = weights / np.sum(np.abs(weights), axis=1, keepdims=True)
        return xg.dot(weights.T)    # (n_halos, n_times)

    # quantile transform x based on training data
    xg = qt_gauss_base(xng, model.x_train)
    mang = xg.dot(weights.T)    # (n_halos, n_times)

    # gaussianize the predicted ma non-gaussian using the train data quantiles.
    # will be from last monte carlo iteration
    # should be fine since we are just using it for ranking predictions,
    # not for actual values.
    mag = qt_gauss_base(mang, model.y_not_gauss_train)

    # invert ma_gauss to data space based on gaussianized y_train.
    ma = qt_inverse_gauss_base(mag, model.y_train)
    return ma


def calc_score(time: np.ndarray, target: float,
               wsum: np.ndarray) -> tuple[np.ndarray, int]:
    """Find index of closest time step to target time and return scores at that time.

    Parameters
    ----------
    time : np.ndarray
        mass accretion history time steps
    target : float
        target time (scale factor or mass)
    wsum : np.ndarray
        scores for each halo as a function of time

    Returns
    -------
    np.ndarray
        scores for the halos at the closest time step to target time
    """
    indx = int((np.abs(time - target)).argmin())
    return wsum[:, indx], indx


def split_mah(mah_dict: dict, scores: np.ndarray, verbose=False):
    """Split mass accretion history into lower and upper percentiles. This function is better for the plotting.

    Parameters
    ----------
    mah_dict : dict
        dictionary of mass accretion histories where keys are halo IDs and values are dataframes
    scores : np.ndarray
        array of scores for each halo
    """
    s25, s50, s75 = np.percentile(scores, [25, 50, 75])
    split_25p = scores < s25
    split_75p = scores > s75
    split_50p = scores == s50
    split_u50p = scores > s50
    split_l50p = scores < s50
    mah_25 = []
    mah_75 = []
    mah_50 = []
    mah_u50 = []
    mah_l50 = []

    for i, (k, madf) in enumerate(mah_dict.items()):
        if split_25p[i]:
            if verbose:
                print(f'{k} in lower 25%')
            mah_25.append(madf['M/M0'].values)
        elif split_75p[i]:
            if verbose:
                print(f'{k} in upper 25%')
            mah_75.append(madf['M/M0'].values)
        elif split_50p[i]:
            if verbose:
                print(f'{k} in 50%')
            mah_50.append(madf['M/M0'].values)

    for i, (k, madf) in enumerate(mah_dict.items()):
        if split_u50p[i]:
            if verbose:
                print(f'{k} in upper 50%')
            mah_u50.append(madf['M/M0'].values)
        elif split_l50p[i]:
            if verbose:
                print(f'{k} in lower 50%')
            mah_l50.append(madf['M/M0'].values)

    mah_25 = np.array(mah_25)  # shape: (N25, Na)
    mah_75 = np.array(mah_75)  # shape: (N75, Na)
    mah_50 = np.array(mah_50)  # shape: (1, Na)
    mah_u50 = np.array(mah_u50)  # shape: (N, Na)
    mah_l50 = np.array(mah_l50)  # shape: (N, Na)

    # Lower 25%
    p50_25 = np.median(mah_25, axis=0)
    p16_25 = np.percentile(mah_25, 16, axis=0, method='linear')
    p84_25 = np.percentile(mah_25, 84, axis=0, method='linear')

    # Upper 25%
    p50_75 = np.median(mah_75, axis=0)
    p16_75 = np.percentile(mah_75, 16, axis=0, method='linear')
    p84_75 = np.percentile(mah_75, 84, axis=0, method='linear')

    if verbose:
        print(split_25p, '\n----------------\n', split_75p)

    return mah_25, mah_50, mah_75, (p16_25, p50_25, p84_25), (p16_75, p50_75, p84_75), mah_u50, mah_l50


def split_mah_idx(mah_dict: dict, scores: np.ndarray, verbose=False):
    """Split mass accretion history into lower and upper percentiles and return halo IDs.

    Parameters
    ----------
    mah_dict : dict
        dictionary of mass accretion histories where keys are halo IDs and values are dataframes
    scores : np.ndarray
        array of scores for each halo
    """
    s25, s50, s75 = np.percentile(scores, [25, 50, 75])
    split_25p = scores < s25
    split_75p = scores > s75
    split_50p = scores == s50
    split_u50p = scores > s50
    split_l50p = scores < s50

    split_dict = {"lower_25": [], "upper_25": [],
                  "median": [], "upper_50": [], "lower_50": []}

    for i, (k, madf) in enumerate(mah_dict.items()):
        if split_25p[i]:
            if verbose:
                print(f'{k} in lower 25%')
            split_dict["lower_25"].append(k)
        elif split_75p[i]:
            if verbose:
                print(f'{k} in upper 25%')
            split_dict["upper_25"].append(k)

        if split_50p[i]:
            if verbose:
                print(f'{k} is median')
            split_dict["median"].append(k)
        elif split_u50p[i]:
            if verbose:
                print(f'{k} in upper 50%')
            split_dict["upper_50"].append(k)
        elif split_l50p[i]:
            if verbose:
                print(f'{k} in lower 50%')
            split_dict["lower_50"].append(k)

    return split_dict

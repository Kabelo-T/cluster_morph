import numpy as np
import pandas as pd

import utils.data as datutils


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


def build_rho(time: np.ndarray, df: pd.DataFrame, param_dict) -> np.ndarray:
    """Build spearman correlation matrix between parameters and mass accretion history, at each time step

    Parameters
    ----------
    time : np.ndarray
        mass accretion history ma or am
    df : pd.DataFrame
       ds or sm parameters
    param_dict : dict
        dictionary of cluster parameters where keys are time steps, values are dataframes of parameters


    Returns
    -------
    np.ndarray
        Spearman correlation matrix
    """
    rho = np.zeros((np.shape(time)[0], np.shape(df)[1]))
    r = 1000
    for i, param in enumerate(df.columns):
        p50 = datutils.get_perc(
            param_dict, param=param, q=50, history='M/M0')
        rho[:, i] = p50

    return rho


def wsum(x, weights: np.ndarray) -> np.ndarray:
    """Weighted sum of cluster parameters.

    Parameters
    ----------
    x : np.ndarray or pd.DataFrame
        ds or sm parameters
    weights : np.ndarray
        multicam training weights or spearman correlation coefficients

    Returns
    -------
    np.ndarray
        scores for each halo as a function of time
    """
    if isinstance(x, pd.DataFrame):
        x = x.to_numpy()

    wsum = x.dot(weights.T)    # (n_halos, n_times)
    return wsum


def calc_score(time: np.ndarray, target: float, wsum: np.ndarray) -> tuple[np.ndarray, int]:
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


def split_mah(mah_dict: dict, scores: np.ndarray):
    """Split mass accretion history into lower and upper percentiles.

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
    mah_25 = []
    mah_75 = []
    mah_50 = []

    for i, (k, madf) in enumerate(mah_dict.items()):
        if split_25p[i]:
            mah_25.append(madf['M/M0'].values)
        elif split_75p[i]:
            mah_75.append(madf['M/M0'].values)
        elif split_50p[i]:
            mah_50.append(madf['M/M0'].values)

    mah_25 = np.array(mah_25)  # shape: (N25, Na)
    mah_75 = np.array(mah_75)  # shape: (N75, Na)
    mah_50 = np.array(mah_50)  # shape: (1, Na)

    # Lower 25%
    p50_25 = np.median(mah_25, axis=0)
    p16_25 = np.percentile(mah_25, 16, axis=0, method='linear')
    p84_25 = np.percentile(mah_25, 84, axis=0, method='linear')

    # Upper 25%
    p50_75 = np.median(mah_75, axis=0)
    p16_75 = np.percentile(mah_75, 16, axis=0, method='linear')
    p84_75 = np.percentile(mah_75, 84, axis=0, method='linear')

    return mah_25, mah_50, mah_75, (p16_25, p50_25, p84_25), (p16_75, p50_75, p84_75)

import os

import numpy as np
import pandas as pd
import astropy.units as u
from astropy.io import fits
from scipy.interpolate import interp1d

import utils.corrs as corutils


def bootstrap(combined_df: pd.DataFrame,
              sample_size: int = None,
              seed: list | int = None) -> list[pd.Series]:
    """Bootstrap the spearman correlation coefficients for the mass accretion 
    histories and the dynamical state parameters (aexp = 1)

    Parameters
    ----------
    combined_df : pd.DataFrame
        Dataframe of mass accretion history and morphology/dynamical state
    sample_size : int, optional
    seed : list|int, optional
    Returns
    -------
    list[pd.Series]
        list of correlations from each samples
    """
    if sample_size is None:
        sample_size = len(combined_df)

    if type(seed) is list:  # if seed is a list, use it to sample
        corrs_list = [
            combined_df.sample(n=sample_size, replace=True,
                               random_state=s).corr(method='spearman')['M/M0']
            for s in seed
        ]
    elif type(seed) is int:
        corrs_list = [
            combined_df.sample(n=sample_size, replace=True,
                               random_state=seed).corr(method='spearman')['M/M0']
            for _ in range(100)
        ]
    else:
        corrs_list = [
            combined_df.sample(n=sample_size, replace=True).corr(
                method='spearman')['M/M0']
            for _ in range(100)
        ]
    return corrs_list


def get_perc(df_dict: dict, param: str, q: int) -> list[float]:
    """Get the qth percentile at each aexp, for the pearson correlations between
    mass accretion histories and dynamical state parameters

    Parameters
    ----------
    df_dict : dict
    param : str
        dynamical state parameter to be selected
    q : int
        the percentile

    Returns
    -------
    percs : list[float]
        qth percentile at each aexp value
    """

    percs = []
    for redshift in df_dict.keys():
        corrs_list = bootstrap(df_dict[redshift])
        param_list = sorted([series[param] for series in corrs_list])
        percs.append(np.percentile(param_list, q=q))
    return percs


def get_percs(df_dict: dict, param:str) -> list[list]:
    p10 = get_perc(df_dict, param=param, q=10)
    p25 = get_perc(df_dict, param=param, q=25)
    p50 = get_perc(df_dict, param=param, q=50)
    p75 = get_perc(df_dict, param=param, q=75)
    p90 = get_perc(df_dict, param=param, q=90)
    return p10, p25, p50, p75, p90


def real2pix(r: u.Quantity, map: np.ndarray, scale=5*u.Mpc) -> int:
    """Convert from physical units to pixels

    Parameters
    ----------
    r : u.Quantity
        radius in Mpc
    map : np.ndarray
    scale : _type_, optional
        size of the map, by default 5*u.Mpc

    Returns
    -------
    radius : int
        the length in pixels
    """
    pixperMpc = map.shape[0]/scale.value
    r = r.to(u.Mpc)
    radius = int(r.value*pixperMpc)
    return radius


def define_ma(df_dict: dict[int, pd.DataFrame]) -> tuple[np.ndarray, np.ndarray]:
    """Sort by strictly increasing aexp and interpolate the mass accretion
    histories to a common set of aexp values.

    Parameters
    ----------
    df_dict : dict[int, pd.DataFrame]
        _description_

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Mass accretion histories and common aexp values
    """
    redshifts = unique_redshifts(df_dict)
    aexp = 1/(1+np.array(redshifts))
    min_a = np.min(aexp)
    max_a = np.max(aexp)
    common_aexp = np.linspace(min_a+0.1, max_a-0.1, 80)

    mah = np.full(shape=(len(df_dict), len(common_aexp)),
                  fill_value=np.nan)

    for i, df in enumerate(df_dict.values()):
        a = df['aexp'].values
        m = df['M/M0'].values

        sort_idx = np.argsort(a)
        a, m = a[sort_idx], m[sort_idx]

        interp_func = interp1d(
            a, m,
            bounds_error=False,
            fill_value=np.nan
        )

        mah[i] = interp_func(common_aexp)
    return mah, common_aexp


def ma(redshift, mah_dict) -> pd.DataFrame:
    mah_df = pd.DataFrame(columns=['ID', 'M/M0'])
    for region in mah_dict.keys():
        row = mah_dict[region].loc[mah_dict[region]
                                   ['Redshift'] == redshift, ['M/M0']]
        row['ID'] = region
        if not row.empty:
            mah_df = pd.concat([mah_df, row], ignore_index=True)
    return mah_df


def unique_redshifts(mah_df_dict: dict[pd.DataFrame]) -> list:
    zs = [x['Redshift'].to_list() for x in mah_df_dict.values()]
    redshifts = []
    for z in zs:
        redshifts.extend(z)

    redshifts = sorted(list(set(redshifts)))

    return redshifts

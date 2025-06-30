import os

import numpy as np
import pandas as pd
import astropy.units as u
from astropy.io import fits
from scipy.interpolate import interp1d

import utils.corrs as corutils


def bootstrap(mah_ds_df: pd.DataFrame,
              sample_size: int = None) -> list[pd.DataFrame]:
    """Bootstrap the spearman correlation coefficients for the mass accretion 
    histories and the dynamical state parameters (aexp = 1)

    Parameters
    ----------
    mah_ds_df : pd.DataFrame
        Dataframe of mass accretion histories and dynamical state params
    sample_size : int, optional

    Returns
    -------
    list[pd.DataFrame]
        list of correlations from each samples
    """
    if not sample_size:
        sample_size = len(mah_ds_df)

    corrs_list = []
    for _ in range(100):
        df = mah_ds_df.sample(n=sample_size, replace=True)
        corrs = df.corr(method='spearman')
        corrs_list.append(corrs['M/M0'])
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
    for k in df_dict.keys():
        corrs_list = bootstrap(df_dict[k])
        param_list = sorted([series[param] for series in corrs_list])
        percs.append(np.percentile(param_list, q=q))
    return percs


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
    radius = r.value*pixperMpc
    return int(radius)


def define_ma(df_dict: dict[pd.DataFrame]):
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
        a_sorted = a[sort_idx]
        m_sorted = m[sort_idx]

        # Only interpolate over increasing a
        interp_func = interp1d(
            a_sorted, m_sorted,
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

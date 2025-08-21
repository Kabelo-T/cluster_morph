import os

import numpy as np
import pandas as pd
import astropy.units as u
from astropy.io import fits
from scipy.interpolate import make_interp_spline

import utils.corrs as corutils


def bootstrap(combined_df: pd.DataFrame,
              sample_size: int = None,
              seed: list | int = None, history: str = 'M/M0') -> list[pd.Series]:
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
                               random_state=s).corr(method='spearman')[history]
            for s in seed
        ]
    elif type(seed) is int:
        corrs_list = [
            combined_df.sample(n=sample_size, replace=True,
                               random_state=seed).corr(method='spearman')[history]
            for _ in range(100)
        ]
    else:
        corrs_list = [
            combined_df.sample(n=sample_size, replace=True).corr(
                method='spearman')[history]
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


def get_percs(df_dict: dict, param: str) -> list[list]:
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


def define_ma(mm0: pd.DataFrame) -> pd.DataFrame:
    """Makes mass accretion history monotonic

    Parameters
    ----------
    mm0 : pd.DataFrame
        mass accretion history

    Returns
    -------
    pd.DataFrame
    """
    m = 0
    n = len(mm0) - 1
    for i in range(n):
        mass = mm0.iloc[i, 0]
        if mass > m:
            m = mass
        elif mass < m:
            mm0.iloc[i, 0] = m
    return mm0


def interp_ma(df_dict: dict[int, pd.DataFrame]) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate the mass accretion histories to a common set of aexp values.

    Parameters
    ----------
    df_dict : dict[int, pd.DataFrame]
        dictionary of each region's mass accretion history

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Mass accretion histories and common aexp values
    """
    redshifts = unique_redshifts(df_dict)
    aexp = 1/(1+np.array(redshifts))
    min_a = np.min(aexp)
    max_a = np.max(aexp)
    num_scales = 80  # 105 is the median, 103 is the mean
    common_aexp = np.linspace(min_a+0.05, max_a-0.05, num_scales)

    mah = np.full(shape=(len(df_dict), num_scales),
                  fill_value=np.nan)

    for i, df in enumerate(df_dict.values()):
        a = df['aexp'].values
        m = df['M/M0'].values
        interp_func = make_interp_spline(a, m, k=3)

        # interp_func = UnivariateSpline(a, m, s=0.5)

        mah[i] = interp_func(common_aexp)
    return mah, common_aexp


def ma_zbin(redshift: float, mah_dict: dict[pd.DataFrame]) -> pd.DataFrame:
    """Gets the m(a) values for all regions at this redshift

    Parameters
    ----------
    redshift : float
    mah_dict : dict[pd.DataFrame]

    Returns
    -------
    pd.DataFrame
    """
    mah_df = pd.DataFrame(columns=['ID', 'M/M0'])
    for region in mah_dict.keys():
        row = mah_dict[region].loc[mah_dict[region]
                                   ['Redshift'] == redshift, ['M/M0']]
        row['ID'] = region
        if not row.empty:
            mah_df = pd.concat([mah_df, row], ignore_index=True)
    return mah_df


def am_mbin(mass: float, mah_dict: dict[pd.DataFrame]) -> pd.DataFrame:
    """Gets the a(m) values for all regions at this mass fraction

    Parameters
    ----------
    mass : float
    mah_dict : dict[pd.DataFrame]

    Returns
    -------
    pd.DataFrame
    """
    mah_df = pd.DataFrame(columns=['ID', 'aexp'])
    for region in mah_dict.keys():
        row = mah_dict[region].loc[mah_dict[region]
                                   ['M/M0'] == mass, ['aexp']]
        row['ID'] = region
        if not row.empty:
            mah_df = pd.concat([mah_df, row], ignore_index=True)
    return mah_df


def ahf_zbin(redshift: float, ahf_dict: dict[pd.DataFrame]) -> pd.DataFrame:
    """Gets the parameters for all regions at this redshift

    Parameters
    ----------
    redshift : float
    ahf_dict : dict[pd.DataFrame]

    Returns
    -------
    pd.DataFrame
    """
    cols = [col for col in ahf_dict[1].columns]
    cols.insert(0, 'ID')
    ahf_df = pd.DataFrame(columns=cols)
    for region in ahf_dict.keys():
        row = ahf_dict[region].loc[ahf_dict[region]['Redshift(0)'] == redshift]
        row['ID'] = region
        if not row.empty:
            ahf_df = pd.concat([ahf_df, row], ignore_index=True)
    return ahf_df


def unique_redshifts(mah_df_dict: dict[pd.DataFrame]) -> list:
    zs = [x['Redshift'].to_list() for x in mah_df_dict.values()]
    redshifts = []
    for z in zs:
        redshifts.extend(z)

    redshifts = sorted(list(set(redshifts)))

    return redshifts


def unique_masses(mah_df_dict: dict[pd.DataFrame]) -> list:
    ms = [x['M/M0'].to_list() for x in mah_df_dict.values()]
    masses = []
    for m in ms:
        masses.extend(m)

    masses = sorted(list(set(masses)))

    return masses

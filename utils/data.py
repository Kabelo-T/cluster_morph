import os

import numpy as np
import pandas as pd
import astropy.units as u
from astropy.io import fits
from scipy.interpolate import interp1d, PchipInterpolator


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


def get_perc(df_dict: dict, param: str, q: int, history: str = 'M/M0') -> list:
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
    for time in df_dict.keys():
        corrs_list = bootstrap(df_dict[time], history=history)
        param_list = sorted([series[param] for series in corrs_list])
        percs.append(np.percentile(param_list, q=q))
    return percs


def get_percs(df_dict: dict, param: str, history: str = 'M/M0') -> tuple:
    p10 = get_perc(df_dict, param=param, q=10, history=history)
    p25 = get_perc(df_dict, param=param, q=25, history=history)
    p50 = get_perc(df_dict, param=param, q=50, history=history)
    p75 = get_perc(df_dict, param=param, q=75, history=history)
    p90 = get_perc(df_dict, param=param, q=90, history=history)
    return p10, p25, p50, p75, p90


def real2pix(r: u.Quantity, map: np.ndarray) -> int:
    """Convert from physical units to pixels

    Parameters
    ----------
    r : u.Quantity
        radius in Mpc
    map : np.ndarray

    Returns
    -------
    radius : int
        the length in pixels
    """
    scale = 5*u.Mpc
    pixperMpc = map.shape[0]/scale.value
    r = r.to(u.Mpc)
    radius = int(r.value*pixperMpc)
    return radius


def interp_ma(df_dict: dict[pd.DataFrame, int], num_scales: int = 100) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate the mass accretion histories to a common set of aexp values.

    Parameters
    ----------
    df_dict : dict[int, pd.DataFrame]
        dictionary of each region's mass accretion history
    num_scales : int, optional
        number of common aexp values, by default 100.
        105 is the median, 103 is the mean, 100 is in multiCAM

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Mass accretion histories and common aexp values
    """
    redshifts = unique_redshifts(df_dict)
    aexp = 1/(1+np.array(redshifts))
    min_a = np.min(aexp)
    common_aexp = np.linspace(min_a, 1., num_scales)

    mah = np.full(shape=(len(df_dict), num_scales),
                  fill_value=0.)
    masses = np.full(shape=(len(df_dict), num_scales),
                     fill_value=0.)

    for i, df in enumerate(df_dict.values()):
        a = df['aexp'].values
        m = df['M/M0'].values
        mass = df['mass'].values
        # interp_func = make_interp_spline(a, m, k=3)
        interp_func = PchipInterpolator(a, m)
        mah[i] = interp_func(common_aexp)

        interp_func = PchipInterpolator(a, mass)
        masses[i] = interp_func(common_aexp)

    return mah, masses, common_aexp


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
    mah_df = pd.DataFrame(columns=['ID', 'M/M0', 'mass'])
    for region in mah_dict.keys():
        row = mah_dict[region].loc[mah_dict[region]
                                   ['Redshift'] == redshift, ['M/M0', 'mass']]
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


def get_am(ma, scales, min_mass_bin, n_bins=100, log_spacing=True):
    """
    1. Inversion is only a well-defined process for monotonic functions, and m(a) for an
    individual halo isn't necessarily monotonic. To solve this, the standard redefinition of a(m0)
    is that it's the first a where m(a) > m0. (This is, for example, how Rockstar defines halfmass
    scales.)

    2. Next, first pick your favorite set of mass bins that you'll evaluate it at. I think
    logarithmic bins spanning 0.01m(a=1) to 1m(a=1) is pretty reasonable, but you should probably
    choose this based on the mass ranges which are the most informative.

    3. Now, for each halo with masses m(a_i), measure M_peak(a_i) = max_j{ m(a_j) | j <= i}.

    4. Remove (a_i, M_peak(a_i)) pairs where M_peak(a_i) = M_peak(a_{i-1}), since this will mess
    up the inversion.

    5. Use scipy.interpolate.interp1d to create a function, f(m), which evaluates a(m).
    For stability, you'll want to run the interpolation on log(a_i) and log(M(a_i)), not a_i and M
    (a_i).

    6. Evaluate f(m) at the mass bins you decided that you liked in step 2. Now you can run your
    pipeline on this, just like you did for m(a).
    """
    assert np.sum(np.isnan(ma)) == 0, (
        "m(a) needs to be filled with `fill_value` previously."
    )

    # 1. + 2.
    if log_spacing:
        mass_bins = np.linspace(np.log(min_mass_bin), np.log(1.0), n_bins)
    else:
        mass_bins = np.log(np.linspace(min_mass_bin, 1.0, n_bins))

    # 3.
    # NOTE: Make function monotonic. We assume start is early -> late ordering.
    # NOTE: ma should not have any cuts.

    # fmax ignores nan's (except beginning ones)
    m_peak = np.fmax.accumulate(ma, axis=1)

    # 4. + 5.
    fs = []
    for i in range(len(m_peak)):  # pylint: disable=consider-using-enumerate
        pairs = [(scales[0], m_peak[i][0])]
        count = 0
        for j in range(1, len(m_peak[i])):
            # keep only pairs that do NOT satisfy (a_{j-1}, Ma_{j-1}) = (a_j, Ma_j)
            if pairs[count][1] != m_peak[i][j]:
                pairs.append((scales[j], m_peak[i][j]))
                count += 1

        assert len(pairs) != 1, (
            "Only 1 pair added, so max reached at a -> 0, impossible."
        )

        _scales = np.array([pair[0] for pair in pairs])
        _m_peaks = np.array([pair[1] for pair in pairs])
        fs.append(
            interp1d(
                np.log(_m_peaks), np.log(_scales), bounds_error=False, fill_value=np.nan
            )
        )

    # 6.
    am = np.array([np.exp(f(mass_bins)) for f in fs])
    return am, np.exp(mass_bins)


def prepare_ma_corrs(mah_dict: dict[int, pd.DataFrame],
                     df0: pd.DataFrame) -> tuple[dict, list]:
    filtered_z = []
    params_dict = {}
    redshifts = unique_redshifts(mah_dict)
    for z in redshifts:
        mah_df = ma_zbin(z, mah_dict)
        if len(mah_df) < 200:
            continue
        mah_df.set_index('ID', inplace=True)
        filtered_z.append(z)
        df = mah_df.merge(df0, on='ID', how='inner')
        params_dict[z] = df

    return params_dict, filtered_z


def prepare_am_corrs(am: np.ndarray, masses: np.ndarray, df0: pd.DataFrame) -> dict:
    params_dict = {}
    mah_df = pd.DataFrame()
    df0 = df0.reset_index()
    df0.drop(columns=['ID'], inplace=True)
    for i, m in enumerate(masses):
        mah_df = pd.DataFrame({'am': am[:, i]})
        df = pd.concat([mah_df, df0], axis=1)
        params_dict[m] = df

    return params_dict

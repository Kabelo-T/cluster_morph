import numpy as np
import pandas as pd
import astropy.units as u
from astropy.io import fits


def print_src_morphs(source_morphs, index=0):

    print('BASIC MEASUREMENTS (NON-PARAMETRIC)')
    print('xc_centroid =', source_morphs[index].xc_centroid)
    print('yc_centroid =', source_morphs[index].yc_centroid)
    print('ellipticity_centroid =', source_morphs[index].ellipticity_centroid)
    print('elongation_centroid =', source_morphs[index].elongation_centroid)
    print('orientation_centroid =', source_morphs[index].orientation_centroid)
    print('xc_asymmetry =', source_morphs[index].xc_asymmetry)
    print('yc_asymmetry =', source_morphs[index].yc_asymmetry)
    print('ellipticity_asymmetry =',
          source_morphs[index].ellipticity_asymmetry)
    print('elongation_asymmetry =', source_morphs[index].elongation_asymmetry)
    print('orientation_asymmetry =',
          source_morphs[index].orientation_asymmetry)
    print('rpetro_circ =', source_morphs[index].rpetro_circ)
    print('rpetro_ellip =', source_morphs[index].rpetro_ellip)
    print('rhalf_circ =', source_morphs[index].rhalf_circ)
    print('rhalf_ellip =', source_morphs[index].rhalf_ellip)
    print('r20 =', source_morphs[index].r20)
    print('r80 =', source_morphs[index].r80)
    print('Gini =', source_morphs[index].gini)
    print('M20 =', source_morphs[index].m20)
    print('F(G, M20) =', source_morphs[index].gini_m20_bulge)
    print('S(G, M20) =', source_morphs[index].gini_m20_merger)
    print('sn_per_pixel =', source_morphs[index].sn_per_pixel)
    print('C =', source_morphs[index].concentration)
    print('A =', source_morphs[index].asymmetry)
    print('S =', source_morphs[index].smoothness)
    print()
    print('SERSIC MODEL')
    print('sersic_amplitude =', source_morphs[index].sersic_amplitude)
    print('sersic_rhalf =', source_morphs[index].sersic_rhalf)
    print('sersic_n =', source_morphs[index].sersic_n)
    print('sersic_xc =', source_morphs[index].sersic_xc)
    print('sersic_yc =', source_morphs[index].sersic_yc)
    print('sersic_ellip =', source_morphs[index].sersic_ellip)
    print('sersic_theta =', source_morphs[index].sersic_theta)
    print('sersic_chi2_dof =', source_morphs[index].sersic_chi2_dof)
    print()
    print('OTHER')
    print('sky_mean =', source_morphs[index].sky_mean)
    print('sky_median =', source_morphs[index].sky_median)
    print('sky_sigma =', source_morphs[index].sky_sigma)
    print('flag =', source_morphs[index].flag)
    print('flag_sersic =', source_morphs[index].flag_sersic)
    return


def create_morph_df(source_morphs, name=None, save=False):

    sources = []
    for id, src in source_morphs:
        sources.append({
            'ID': id,
            'xc_centroid': src.xc_centroid,
            'yc_centroid': src.yc_centroid,
            'ellipticity_asymmetry': src.ellipticity_asymmetry,
            'ellipticity_centroid': src.ellipticity_centroid,
            'elongation_asymmetry': src.elongation_asymmetry,
            'elongation_centroid': src.elongation_centroid,
            'orientation_centroid': src.orientation_centroid,
            'xc_asymmetry': src.xc_asymmetry,
            'yc_asymmetry': src.yc_asymmetry,
            'ellipticity_asymmetry': src.ellipticity_asymmetry,
            'elongation_asymmetry': src.elongation_asymmetry,
            'orientation_asymmetry': src.orientation_asymmetry,
            'rpetro_circ': src.rpetro_circ,
            'rpetro_ellip': src.rpetro_ellip,
            'rhalf_circ': src.rhalf_circ,
            'rhalf_ellip': src.rhalf_ellip,
            'r20': src.r20,
            'r50': src.r50,
            'r80': src.r80,
            'Gini': src.gini,
            'M20': src.m20,
            'F(G, M20)': src.gini_m20_bulge,
            'S(G, M20)': src.gini_m20_merger,
            'sn_per_pixel': src.sn_per_pixel,
            'C': src.concentration,
            'A': src.asymmetry,
            'S': src.smoothness,
            'sersic_amplitude': src.sersic_amplitude,
            'sersic_rhalf': src.sersic_rhalf,
            'sersic_n': src.sersic_n,
            'sersic_xc': src.sersic_xc,
            'sersic_yc': src.sersic_yc,
            'sersic_ellip': src.sersic_ellip,
            'sersic_theta': src.sersic_theta,
            'sersic_chi2_dof': src.sersic_chi2_dof,
            'flag': src.flag,
            'flag_sersic': src.flag_sersic,
            'flux_circ': src.flux_circ,
            'flux_ellip': src.flux_ellip,
            'runtime (s)': src.runtime
        })

    sources = pd.DataFrame(sources)
    sources.set_index('ID', inplace=True)
    if save:
        if name is not None:
            sources.to_csv(name)
        else:
            sources.to_csv('source_morphs.csv')
    return sources


def get_mah(file: str) -> pd.DataFrame:
    """Load the mass accretion histories for a cluster.

    Parameters
    ----------
    file : str

    Returns
    -------
    mm0 : pd.DataFrame
        mass accretion history as M(z)/M(z=0)
    """
    if '.dat' not in file:
        return
    mah_df = pd.read_csv(file, sep=r'\s+', index_col=False)
    mm0 = mah_df['Mvir(4)'].values/mah_df['Mvir(4)'][0]
    mm0 = pd.DataFrame(mm0)
    mm0.rename(columns={0: 'M/M0'}, inplace=True)
    mm0['Redshift'] = mah_df['Redshift(0)']
    mm0['aexp'] = 1 / (1+mm0['Redshift'])
    return mm0


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


def get_perc(mah_ds_dict: dict, param: str, q: int) -> list[float]:
    """Get the qth percentile at each aexp, for the pearson correlations between
    mass accretion histories and dynamical state parameters

    Parameters
    ----------
    mah_ds_dict : dict
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
    for k in mah_ds_dict.keys():
        corrs_list = bootstrap(mah_ds_dict[k])
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
    radius = r.value*pixperMpc
    return int(radius)


def find_id(file: str) -> int:
    """Find halo ID for mass accretion histories

    Parameters
    ----------
    file : str
        file name

    Returns
    -------
    id  
    """

    ind = file.find('_0')
    sub = file[ind+1:ind+5]
    id = int(sub.lstrip('0'))
    return id


def load_map(file, map_dir):
    if '.fits' in file:
        map_file = map_dir + file

    try:
        map = fits.open(map_file)
    except FileNotFoundError:
        return None

    map = fits.open(map_file)
    map = map[0].data
    return map

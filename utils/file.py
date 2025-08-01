import os

import pandas as pd
from astropy.io import fits

import utils.data as datutils


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


def find_id(file: str) -> int:
    """Find halo ID for mass accretion histories

    Parameters
    ----------
    file : str
        file name

    Returns
    -------
    idx  
    """

    ind = file.find('_0')
    sub = file[ind+1:ind+5]
    idx = int(sub.lstrip('0'))
    return idx


def create_morph_df(source_morphs, name=None, save=False):

    sources = []
    for idx, src in source_morphs:
        sources.append({
            'ID': idx,
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
    sources.sort_index(inplace=True)
    if save:
        if name is not None:
            sources.to_csv(name)
        else:
            sources.to_csv('source_morphs.csv')
    return sources


def load_map(file, map_dir):
    if '.fits' in file:
        map_file = map_dir + file

    try:
        map = fits.open(map_file)
    except FileNotFoundError:
        print('File not found.')
        return None

    map = fits.open(map_file)
    map = map[0].data
    return map


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


def get_mah_all(mah_dir: str = 'data/gadgetx3k/AHFHaloHistory/') -> dict:
    df_dict = {}
    for f in sorted(os.listdir(mah_dir)):
        file = mah_dir + f
        mm0 = get_mah(file)
        idx = find_id(file)
        df_dict[idx] = mm0

    mah, common_aexp = datutils.define_ma(df_dict)
    mah_df_dict = {}
    for i, key in enumerate(df_dict.keys()):
        df = pd.DataFrame(columns=['M/M0', 'Redshift', 'aexp'])
        df['M/M0'] = mah[i]
        df['aexp'] = common_aexp
        df['Redshift'] = (1/df['aexp']) - 1
        mah_df_dict[key] = df

    return mah_df_dict


def get_morphologies(sm_dir: str) -> pd.DataFrame:
    sm_df = pd.read_csv(sm_dir)
    sm_df.set_index('ID', inplace=True)
    sm_df.drop(columns=['flag', 'flag_sersic', 'flux_circ', 'flux_ellip',
                        'sn_per_pixel', 'runtime (s)'], inplace=True)
    return sm_df


def get_ds_theory_today(file_path: str = 'data/gadgetx3k/GadgetX-DS-theory-snap-128.txt'):
    ds_z0 = {}
    with open(file_path) as f:
        for i, x in enumerate(f):
            if i == 0:
                continue
            k, v = list(map(int, x.split()))
            ds_z0[k] = v
    return ds_z0


def get_ds(file: str = 'data/gadgetx3k/G3X_progenitors/DS_G3X_snap_128_center-cluster_progenitors.txt') -> pd.DataFrame:
    dsdf = pd.read_csv(file, sep=r'\s+', header=0)
    int_columns = [0, 1, 2, 7]
    column_names = dsdf.columns
    for idx in range(len(column_names)):
        col_name = column_names[idx]
        if idx in int_columns:
            dsdf[col_name] = dsdf[col_name].astype(int)
        else:
            dsdf[col_name] = dsdf[col_name].astype(float)

    dsdf.drop(columns=['Hid[1]', 'DS_200[2]', 'DS_500[7]'], inplace=True)
    dsdf.rename(columns={'rID[0]': 'ID'}, inplace=True)
    dsdf.set_index('ID', inplace=True)
    return dsdf

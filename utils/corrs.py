import os

import pandas as pd
import numpy as np
import utils.data as datutils


def build_ma_corrs(mah_df_dict: dict[pd.DataFrame],
                   df0: pd.DataFrame) -> list:
    corrs_list = []
    filtered_z = []
    params_dict = {}
    redshifts = datutils.unique_redshifts(mah_df_dict)
    for z in redshifts:
        mah_df = datutils.ma_zbin(z, mah_df_dict)
        if len(mah_df) < 200:
            continue

        mah_df.set_index('ID', inplace=True)
        filtered_z.append(z)

        df = mah_df.merge(df0, on='ID', how='inner')
        params_dict[z] = df
        corrs = df.corr(method='spearman')
        corrs_list.append(corrs)

    return params_dict, corrs_list, filtered_z


def build_am_corrs(am: np.ndarray, mbins: np.ndarray, sm_df: pd.DataFrame,
                   mah_dir: str = 'data/gadgetx3k_20/AHFHaloHistory') -> list:
    mah_files = sorted(os.listdir(mah_dir))
    ids = [datutils.find_id(f) for f in mah_files]
    am_df = pd.DataFrame(am)
    am_df['ID'] = ids
    am_df.set_index('ID', inplace=True)

    filtered_mbins = []
    corrs_list = []
    ind_cols = am_df.columns
    for col in ind_cols:
        mah_df = am_df[col].dropna()
        if len(mah_df) < 100:
            continue

        filtered_mbins.append(mbins[col])

        df = pd.concat([mah_df, sm_df], join='inner', axis=1)

        corrs = df.corr(method='spearman')
        corrs_list.append(corrs)
    return corrs_list, filtered_mbins

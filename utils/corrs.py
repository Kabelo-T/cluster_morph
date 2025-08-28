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
    for i, z in enumerate(redshifts):
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


def build_am_corrs(am: np.ndarray, masses: np.ndarray, df0: pd.DataFrame) -> list:
    corrs_list = []
    params_dict = {}
    mah_df = pd.DataFrame()
    df0 = df0.reset_index()
    df0.drop(columns=['ID'], inplace=True)
    for i, m in enumerate(masses):
        mah_df = pd.DataFrame({'am': am[:, i]})
        df = pd.concat([mah_df, df0], axis=1)
        params_dict[m] = df
        corrs = df.corr(method='spearman')
        corrs_list.append(corrs)

    return params_dict, corrs_list

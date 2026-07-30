import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.cosmology import Planck13

import utils.file as futils
import utils.data as datutils
import utils.plots as plots
import utils.corrs as corrs
import argparse
import warnings
warnings.filterwarnings('ignore')

MAH_DIR = 'data/AHF_HaloHistory'
DS_PARAMS = ['eta_200[3]', 'delta_200[4]', 'fm_200[5]', 'fm2_200[6]', '3d',
             'r_sp', 'r_trunc']
DS_LABELS = [r'$\eta_{200c}$', r'$\Delta_{200c}$',
             r'$f_{m, 200c}$', r'$f_{m2, 200c}$', r'$m_{14, 3D}$',
             'Splashback', 'Truncation']
SM_PARAMS = ['rhalf_circ', 'C', 'sersic_amplitude', 'A', 'm14', 'core_C']
SM_PARAM_LABELS = [r'Half-light Radius, SM$_{50kpc,1Mpc}$: $r_{1/2}$',
                    r'Concentration, SM$_{50kpc,1Mpc}$: $C_{50\mathrm{kpc},1\mathrm{Mpc}}$',
                    r'Sersic Amplitude, SM$_{50kpc,1Mpc}$: $SA_{50\mathrm{kpc},1\mathrm{Mpc}}$',
                    r'Asymmetry, SM$_{50kpc,1Mpc}$: $A_{50\mathrm{kpc},1\mathrm{Mpc}}$',
                    r'Magnitude Gap: $m_{14}$',
                    r'Core Concentration, SM$_{<30kpc}$: $C_{<30\mathrm{kpc}}$']
M14 = futils.get_m14()

def valid_ids(ixy, iyz, izx):
    idxy = {int(k) for k in ixy.index}
    idyz = {int(k) for k in iyz.index}
    idzx = {int(k) for k in izx.index}
    halo_ids = list(idxy & idyz & idzx)

    return halo_ids

def load_sm():
    ixy = futils.get_morph(sm_dir='results/xy/r_0.03Mpc_300.csv')
    iyz = futils.get_morph(sm_dir='results/yz/r_0.03Mpc_286.csv')
    izx = futils.get_morph(sm_dir='results/zx/r_0.03Mpc_297.csv')
    halo_ids = valid_ids(ixy, iyz, izx)
    inner_df = pd.concat([ixy.loc[halo_ids], iyz.loc[halo_ids], izx.loc[halo_ids]])
    iconc = inner_df['C']
    iconc.name = 'core_C'
    m14 = M14.loc[halo_ids]
    m14_stacked = np.hstack([m14['proj_x'], m14['proj_y'], m14['proj_z']])
    smdf = futils.get_morphologies(halo_ids)
    ahf = futils.get_ahf_all()
    mass = np.array([ahf[halo_id]['Mvir(4)'][ahf[halo_id]['Redshift(0)'] == 0].values[0]
                     for halo_id in halo_ids])
    logmass = np.log10(mass)
    logmass_stacked = np.hstack([logmass]*3)  # because we have 3 projections
    lmass_df = pd.DataFrame({'logmass': logmass_stacked})
    n_halos = len(halo_ids)
    assert m14_stacked.shape[0] == 3 * n_halos
    assert smdf.shape[0] == 3 * n_halos
    assert iconc.shape[0] == 3 * n_halos
    m14_xyz = pd.Series(m14_stacked)
    df = pd.concat([smdf.reset_index(), m14_xyz.reset_index(drop=True),
                    iconc.reset_index(drop=True)], axis=1)
    df.rename(columns={0: 'm14'}, inplace=True)

    return df, halo_ids

def load(features):
    if features == 'sm':
        df, halo_ids = load_sm()
    else:
        dsdf = futils.get_ds()
        dsdf.drop(columns=['eta_500[8]', 'delta_500[9]',
                 'fm_500[10]', 'fm2_500[11]'], inplace=True)
        df = pd.concat([dsdf, M14['3d']], axis=1, join='inner')
        df.sort_index(inplace=True)
        halo_ids = list(df.index)

    mah_dict = futils.get_mah_all(halo_ids=halo_ids)
    ma, _, aexp_bins = datutils.interp_ma(mah_dict)
    am, mass_bins = datutils.build_am(ma=ma, scales=aexp_bins,
                                      min_mass_bin=0.01, log_spacing=True)
    return mah_dict, ma, am, aexp_bins, mass_bins, df

def calc_lbt(aexp):
    redshifts = 1/aexp -1
    lookback_time = Planck13.lookback_time(redshifts).value  # in Gyr

    return lookback_time


SM_LABELS = [r'$\mathrm{SM} + m_{14} + \mathrm{Core\,Conc.}$',
             r'$\mathrm{SM} + m_{14}$', r'$\mathrm{SM}$', r'$A$']

DS_CURVE_LABELS = [' + '.join(DS_LABELS[:5]), ' + '.join(DS_LABELS[1:5]),
                    ' + '.join(DS_LABELS[2:5]), DS_LABELS[4],
                    DS_LABELS[5], DS_LABELS[6]]

CURVE_COLORS = ['k', '#56B4E9', '#F0E442', '#009E73', '#E69F00', '#CC79A7']
CURVE_FILL = [True, False, False, True, True, True]

def add_lbt_twiny(axs, xticks, xlim):
    ax2 = axs.twiny()
    ax2.set_xlim(xlim)
    a_ticks = xticks[(xticks > 0) & (xticks <= xlim[1])]
    ax2.set_xticks(a_ticks)
    ax2.set_xticklabels([f"{t:.1f}" for t in calc_lbt(a_ticks)])
    ax2.set_xlabel("Lookback Time (Gyr)", size=13)

def plot_preds(axs, curves, tbins, labels, colors=CURVE_COLORS, fill=CURVE_FILL):
    xticks = np.arange(0, 1.1, 0.1)
    xlim = (0, 1.01)
    ylim = (0., 0.8)

    for curve, label, color, fl in zip(curves, labels, colors, fill):
        axs.plot(tbins, np.abs(curve[1]), color=color, label=label)
        if fl:
            axs.fill_between(tbins, np.abs(curve[0]), np.abs(curve[2]),
                              color=color, alpha=0.2)

    axs.set_xlabel('Scale Factor a = 1/(1+z)')
    axs.set_xticks(xticks)
    axs.set_xlim(xlim)
    axs.set_ylabel(r'$\rho_s (m(a)_{test}, m(a)_{pred})$')
    axs.set_ylim(ylim)
    axs.grid()
    axs.legend()

    add_lbt_twiny(axs, xticks, xlim)

def plot_feature_corrs(axs, corr_dict, params, labels, tbins, sm=True):
    xticks = np.arange(0, 1.1, 0.1)
    xlim = (0, 1.01)

    plots.plot_corrs(corr_dict, params=params, labels=labels,
                      time=tbins[::-1], axs=axs, sm=sm, history='M/M0')
    axs.set_xticks(xticks)
    axs.set_xlim(xlim)

    add_lbt_twiny(axs, xticks, xlim)

def sm_predict(features):
    mah_dict, mah, _, tbins, _,  df = load(features)
    corr_dict, _ = datutils.prepare_ma_corrs(mah_dict, df)
    mah = np.vstack([mah]*3)  # because we have 3 projections
    curves = []
    y = mah

    x = np.vstack([df['rhalf_circ'], df['C'], df['A'],
           df['sersic_amplitude'], df['m14'], df['core_C']]).T
    _, p25, p50, p75, _, _, _ = corrs.multicorr(x, y, proj=True)
    curves.append(np.array([p25, p50, p75]))

    x = np.vstack([df['rhalf_circ'], df['C'], df['A'],
           df['sersic_amplitude'], df['m14']]).T
    _, p25, p50, p75, _, _, _ = corrs.multicorr(x, y, proj=True)
    curves.append(np.array([p25, p50, p75]))

    x = np.vstack([df['rhalf_circ'], df['C'], df['A'],
           df['sersic_amplitude']]).T
    _, p25, p50, p75, _, _, _ = corrs.multicorr(x, y, proj=True)
    curves.append(np.array([p25, p50, p75]))

    x = np.vstack([df['A']]).T
    _, p25, p50, p75, _, _, _ = corrs.multicorr(x, y, proj=True)
    curves.append(np.array([p25, p50, p75]))

    curves = np.array(curves)
    return mah_dict, curves, tbins, corr_dict


def ds_predict(features):
    mah_dict, mah, _, tbins, _,  df = load(features)
    corr_dict, _ = datutils.prepare_ma_corrs(mah_dict, df)
    curves = []
    y = mah

    x = np.vstack([df['eta_200[3]'], df['delta_200[4]'], df['fm_200[5]'],
                   df['fm2_200[6]'], df['3d']]).T
    _, p25, p50, p75, _, _, _ = corrs.multicorr(x, y, proj=False)
    curves.append(np.array([p25, p50, p75]))

    x = np.vstack([df['delta_200[4]'], df['fm_200[5]'],
                   df['fm2_200[6]'], df['3d']]).T
    _, p25, p50, p75, _, _, _ = corrs.multicorr(x, y, proj=False)
    curves.append(np.array([p25, p50, p75]))

    x = np.vstack([df['fm_200[5]'], df['fm2_200[6]'], df['3d']]).T
    _, p25, p50, p75, _, _, _ = corrs.multicorr(x, y, proj=False)
    curves.append(np.array([p25, p50, p75]))

    x = np.vstack([df['3d']]).T
    _, p25, p50, p75, _, _, _ = corrs.multicorr(x, y, proj=False)
    curves.append(np.array([p25, p50, p75]))

    x = np.vstack([df['r_sp']]).T
    _, p25, p50, p75, _, _, _ = corrs.multicorr(x, y, proj=False)
    curves.append(np.array([p25, p50, p75]))

    x = np.vstack([df['r_trunc']]).T
    _, p25, p50, p75, _, _, _ = corrs.multicorr(x, y, proj=False)
    curves.append(np.array([p25, p50, p75]))


    curves = np.array(curves)
    return mah_dict, curves, tbins, corr_dict


def predict(features):
    if features == 'sm':
        mah_dict, curves, tbins, corr_dict = sm_predict(features)
        labels, params, param_labels, sm_flag = SM_LABELS, SM_PARAMS, SM_PARAM_LABELS, True
    else:
        mah_dict, curves, tbins, corr_dict = ds_predict(features)
        labels, params, param_labels, sm_flag = DS_CURVE_LABELS, DS_PARAMS, DS_LABELS, False

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(16, 6), constrained_layout=True)
    plot_feature_corrs(axs[0], corr_dict, params, param_labels, tbins, sm=sm_flag)
    plot_preds(axs[1], curves, tbins, labels)
    plt.show()
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--features', type=str,
        default='sm', help="features to make predictions, sm or ds")
    args = parser.parse_args()
    predict(args.features)

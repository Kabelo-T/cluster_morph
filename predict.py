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

MAH_DIR = 'data/gadgetx3k/GadgetX'
DS_PARAMS = ['eta_200[3]', 'delta_200[4]', 'fm_200[5]', 'fm2_200[6]', '3d']
DS_LABELS = [r'$\eta_{200c}$', r'$\Delta_{200c}$',
          r'$f_{m, 200c}$', r'$f_{m2, 200c}$', r'$m_{14, 3D}$']
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
    df = pd.concat([smdf.reset_index(), m14_xyz.reset_index(drop=True), iconc.reset_index(drop=True)], axis=1)
    df.rename(columns={0: 'm14'}, inplace=True)

    return df, halo_ids

def load(features):
    if features == 'sm':
        df, halo_ids = load_sm()
    else:
        dsdf = futils.get_ds()
        dsdf.drop(columns=['eta_500[8]', 'delta_500[9]',
                 'fm_500[10]', 'fm2_500[11]'], inplace=True)
        df = pd.concat([dsdf, M14['3d']], axis=1)

    mah_dict = futils.get_mah_all(halo_ids=halo_ids)
    ma, _, aexp_bins = datutils.interp_ma(mah_dict)
    am, mass_bins = datutils.build_am(ma=ma, scales=aexp_bins,
                                      min_mass_bin=0.01, log_spacing=True)
    return mah_dict, ma, am, aexp_bins, mass_bins, df

def calc_lbt(aexp):
    redshifts = 1/aexp -1
    lookback_time = Planck13.lookback_time(redshifts).value  # in Gyr

    return lookback_time


def plot_preds(axs, curves, tbins):
    xticks = np.arange(0, 1.1, 0.1)
    xlim = (0, 1.01)
    ylim = (0., 0.7)
    axs.plot(tbins, curves[0, 1], color='k',
                label=r'$\mathrm{SM} + m_{14} + \mathrm{Core\,Conc.}$') 
    axs.fill_between(tbins, curves[0, 0], curves[0, 2], color='k', alpha=0.2)
    
    axs.plot(tbins, np.abs(curves[1, 1]), color='#56B4E9',
                label=r'$\mathrm{SM} + m_{14}$')
    axs.plot(tbins, np.abs(curves[2, 1]), color='#F0E442',
                label=r'$\mathrm{SM}$')
    
    axs.plot(tbins, np.abs(curves[3, 1]), color='#D55E00',
                label=r'$A + \mathrm{Core\,Conc.} + m_{14}$')
    
    axs.plot(tbins, np.abs(curves[4, 1]), color='#009E73',
                label=r'$A + \mathrm{Core\,Conc.}$')
    
    axs.plot(tbins, np.abs(curves[5, 1]), color='#009E73',
                label=r'$A$')
    axs.fill_between(tbins, np.abs(curves[5, 0]), np.abs(curves[5, 2]), color='#009E73', alpha=0.2)

    axs.set_xlabel('Scale Factor a = 1/(1+z)')
    axs.set_xticks(xticks)
    axs.set_xlim(xlim)
    axs.set_ylabel(r'$\rho_s (m(a)_{test}, m(a)_{pred})$')
    axs.set_ylim(ylim)
    axs.grid()
    axs.legend()

    ax2 = axs.twiny()
    ax2.set_xlim(xlim)
    a_ticks = xticks[(xticks > 0) & (xticks <= xlim[1])]
    ax2.set_xticks(a_ticks)
    ax2.set_xticklabels([f"{t:.1f}" for t in calc_lbt(a_ticks)])
    ax2.set_xlabel("Lookback Time (Gyr)", size=13)    

def sm_predict(features):
    mah_dict, mah, _, tbins, _,  df = load(features)
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

    x = np.vstack([df['A'], df['m14'], df['core_C']]).T
    _, p25, p50, p75, _, _, _ = corrs.multicorr(x, y, proj=True)
    curves.append(np.array([p25, p50, p75]))

    x = np.vstack([df['A'], df['core_C']]).T
    _, p25, p50, p75, _, _, _ = corrs.multicorr(x, y, proj=True)
    curves.append(np.array([p25, p50, p75]))

    x = np.vstack([df['A']]).T
    _, p25, p50, p75, _, _, _ = corrs.multicorr(x, y, proj=True)
    curves.append(np.array([p25, p50, p75]))

    curves = np.array(curves)
    return mah_dict, curves, tbins


def predict(features):
    if features == 'sm':
        mah_dict, curves, tbins = sm_predict(features)
        
    fig, axs = plt.subplots()
    plot_preds(axs, curves, tbins)
    plt.show()
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--features', type=str,
        default='sm', help="features to make predictions, sm or ds")
    args = parser.parse_args()
    predict(args.features)

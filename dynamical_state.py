import argparse
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from astropy.cosmology import Planck18

import utils.file as futils
import utils.data as datutils
import utils.plots as plots

warnings.filterwarnings('ignore')

RADIUS_COLUMNS = {
    'r200': {'eta': 'eta_200[3]', 'delta': 'delta_200[4]',
             'fm': 'fm_200[5]', 'fm2': 'fm2_200[6]'},
    'r500': {'eta': 'eta_500[8]', 'delta': 'delta_500[9]',
             'fm': 'fm_500[10]', 'fm2': 'fm2_500[11]'},
}
RADIUS_LABELS = {
    'r200': {'eta': r'$\eta_{200c}$', 'delta': r'$\Delta_{200c}$',
             'fm': r'$f_{m, 200c}$', 'fm2': r'$f_{m2, 200c}$', 'suffix': '200c'},
    'r500': {'eta': r'$\eta_{500c}$', 'delta': r'$\Delta_{500c}$',
             'fm': r'$f_{m, 500c}$', 'fm2': r'$f_{m2, 500c}$', 'suffix': '500c'},
}


def dynamical_state(r500=False, save=False):
    radius = 'r500' if r500 else 'r200'
    cols = RADIUS_COLUMNS[radius]
    labels = RADIUS_LABELS[radius]
    other_cols = RADIUS_COLUMNS['r200' if r500 else 'r500']
    drop_cols = ['snap'] + list(other_cols.values())

    redshift_list = pd.read_csv('data/redshift_list.txt', sep=r'\s+')
    # for some reason I am only getting values after a>0.25
    scales = redshift_list[66:]['a'].values
    redshifts = redshift_list[66:]['z'].values
    lookback_time = Planck18.lookback_time(redshifts).value  # in Gyr

    dsdict = futils.get_ds_all()
    mah_dict = futils.get_mah_all()
    ma, _, aexp_bins = datutils.interp_ma(mah_dict, num_scales=97)

    rds_dict = {}
    for i, (snap, df) in enumerate(dsdict.items()):
        reduced = df.loc[mah_dict.keys()].drop(columns=drop_cols)
        reduced['m(a)'] = ma[:, i]
        rds_dict[snap] = reduced

    eta_corr, delta_corr, fm_corr, fm2_corr = [], [], [], []
    for i, df in enumerate(rds_dict.values()):
        m_a = df['m(a)'].values
        for j, df1 in enumerate(rds_dict.values()):
            eta_corr.append(
                spearmanr(df1[cols['eta']].values, m_a, nan_policy='omit')[0])
            delta_corr.append(
                spearmanr(df1[cols['delta']].values, m_a, nan_policy='omit')[0])
            fm_corr.append(
                spearmanr(df1[cols['fm']].values, m_a, nan_policy='omit')[0])
            fm2_corr.append(
                spearmanr(df1[cols['fm2']].values, m_a, nan_policy='omit')[0])

    n_scales = len(rds_dict)
    eta_corr = np.array(eta_corr).reshape(n_scales, n_scales)
    delta_corr = np.array(delta_corr).reshape(n_scales, n_scales)
    fm_corr = np.array(fm_corr).reshape(n_scales, n_scales)
    fm2_corr = np.array(fm2_corr).reshape(n_scales, n_scales)

    ma_corrs, _ = spearmanr(ma, axis=0)

    fig, axs = plt.subplots(2, 2, figsize=(14, 12))

    plots.plot_2dcorr(ma_corrs, scales=scales, xlabel=r'$a_i$',
                      ylabel=r'$a_j$',
                      ax=axs[0, 0])
    axs[0, 0].annotate(r'$\rho_s (m(a_i), m(a_j))$', xy=(0.03, 0.93), xycoords='axes fraction',
                       fontsize=16)

    plots.plot_2dcorr(eta_corr, scales=scales, xlabel=r'$a_i$',
                      ylabel=r'$a_j$',
                      ax=axs[0, 1])
    axs[0, 1].annotate(fr'$\rho_s ({labels["eta"][1:-1]}(a_i), m(a_j))$', xy=(0.03, 0.93),
                       xycoords='axes fraction', fontsize=16)

    plots.plot_2dcorr(delta_corr, scales=scales, xlabel=r'$a_i$',
                      ylabel=r'$a_j$',
                      ax=axs[1, 0])
    axs[1, 0].annotate(fr'$\rho_s ({labels["delta"][1:-1]}(a_i), m(a_j))$', xy=(0.03, 0.93),
                       xycoords='axes fraction', fontsize=16)
    # Get handle to the image from the last plot
    im, boundaries = plots.plot_2dcorr(fm2_corr, scales=scales, xlabel=r'$a_i$',
                                       ylabel=r'$a_j$',
                                       ax=axs[1, 1])
    axs[1, 1].annotate(fr'$\rho_s ({labels["fm2"][1:-1]}(a_i), m(a_j))$', xy=(0.03, 0.93),
                       xycoords='axes fraction', fontsize=16)

    # Add upper x-axes with lookback time labels using secondary_xaxis
    # map scale (a) -> lookback time (td) and inverse using interpolation

    def scale_to_lb(s): return np.interp(s, scales, lookback_time)  # type: ignore

    def lb_to_scale(t): return np.interp(
        t,
        lookback_time[::-1],
        scales[::-1]
    )   # type: ignore

    for ax in axs.ravel():
        sec = ax.secondary_xaxis('top', functions=(scale_to_lb, lb_to_scale))
        xmin, xmax = ax.get_xlim()
        a_ticks = ax.get_xticks()
        a_ticks = a_ticks[(a_ticks >= xmin) & (a_ticks <= xmax)]
        lb_ticks = scale_to_lb(a_ticks)

        sec.set_xticks(lb_ticks)
        sec.set_xticklabels([f"{t:.1f}" for t in lb_ticks])
        sec.set_xlabel("Lookback Time (Gyr)", size=13)

    cbar = fig.colorbar(im, ax=axs.ravel().tolist(),
                        boundaries=boundaries, ticks=boundaries,
                        location='right', fraction=0.05)

    cbar.ax.set_yticklabels([f'{b:.1f}' for b in boundaries])
    cbar.set_label(
        fr'Correlation coefficient  $\rho_s (DS_{{{labels["suffix"]}}}(a_i), m(a_j))$', fontsize=20)

    plt.tight_layout(rect=[0, 0, 0.84, 1])  # leave space on right for colorbar
    if save:
        plt.savefig(f'plots/ds_{labels["suffix"]}_ma.pdf')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--r500', action='store_true',
                        help="Use r500 dynamical-state features instead of r200")
    parser.add_argument('-s', '--save', action='store_true',
                        help="Save plot")
    args = parser.parse_args()
    dynamical_state(r500=args.r500, save=args.save)

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
import pandas as pd
from astropy.cosmology import Planck18

import utils.file as futils
import utils.data as datutils
import utils.plots as plots

import warnings

warnings.filterwarnings('ignore')

redshift_list = pd.read_csv('data/redshift_list.txt', sep=r'\s+')
# for some reason I am only getting values after a>0.25
scales = redshift_list[66:]['a'].values
redshifts = redshift_list[66:]['z'].values
lookback_time = Planck18.lookback_time(redshifts).value  # in Gyr

dsdict = futils.get_ds_all()

dsdict_200 = {}
dsdict_500 = {}
for snap, df in dsdict.items():
    dsdict_200[snap] = df.drop(columns=['snap', 'eta_500[8]',
                                        'delta_500[9]', 'fm_500[10]', 'fm2_500[11]'])
    dsdict_500[snap] = df.drop(columns=['snap', 'eta_200[3]',
                                        'delta_200[4]', 'fm_200[5]', 'fm2_200[6]'])

mah_dict = futils.get_mah_all()
ma, _, aexp_bins = datutils.interp_ma(mah_dict, num_scales=97)
reduced_ds_dict = {}
for k, df in dsdict.items():
    reduced_ds_dict[k] = df.loc[mah_dict.keys()]

rdsdict_200 = {}
rdsdict_500 = {}
for i, (snap, df) in enumerate(reduced_ds_dict.items()):
    rdsdict_200[snap] = df.drop(columns=['snap', 'eta_500[8]',
                                         'delta_500[9]', 'fm_500[10]', 'fm2_500[11]'])
    rdsdict_200[snap]['m(a)'] = ma[:, i]

    rdsdict_500[snap] = df.drop(columns=['snap', 'eta_200[3]',
                                         'delta_200[4]', 'fm_200[5]', 'fm2_200[6]'])
    rdsdict_500[snap]['m(a)'] = ma[:, i]

eta200c, delta200c, fm200c, fm2200c = [], [], [], []
for i, df in enumerate(rdsdict_200.values()):
    m_a = df['m(a)'].values
    for j, df1 in enumerate(rdsdict_200.values()):
        ma_eta, _ = spearmanr(df1['eta_200[3]'].values, m_a, nan_policy='omit')
        eta200c.append(ma_eta)

        ma_delta, _ = spearmanr(
            df1['delta_200[4]'].values, m_a, nan_policy='omit')
        delta200c.append(ma_delta)

        ma_fm, _ = spearmanr(df1['fm_200[5]'].values, m_a, nan_policy='omit')
        fm200c.append(ma_fm)

        ma_fm2200, _ = spearmanr(
            df1['fm2_200[6]'].values, m_a, nan_policy='omit')
        fm2200c.append(ma_fm2200)

eta200c = np.array(eta200c).reshape(97, 97)
delta200c = np.array(delta200c).reshape(97, 97)
fm200c = np.array(fm200c).reshape(97, 97)
fm2200c = np.array(fm2200c).reshape(97, 97)

ma_corrs, _ = spearmanr(ma, axis=0)

fig, axs = plt.subplots(2, 2, figsize=(14, 12))

plots.plot_2dcorr(ma_corrs, scales=scales, xlabel=r'$a_i$',
                  ylabel=r'$a_j$',
                  ax=axs[0, 0])
axs[0, 0].annotate(r'$\rho_s (m(a_i), m(a_j))$', xy=(0.03, 0.93), xycoords='axes fraction',
                   fontsize=16)

plots.plot_2dcorr(eta200c, scales=scales, xlabel=r'$a_i$',
                  ylabel=r'$a_j$',
                  ax=axs[0, 1])
axs[0, 1].annotate(r'$\rho_s (\eta_{200c}(a_i), m(a_j))$', xy=(0.03, 0.93), xycoords='axes fraction',
                   fontsize=16)

plots.plot_2dcorr(delta200c, scales=scales, xlabel=r'$a_i$',
                  ylabel=r'$a_j$',
                  ax=axs[1, 0])
axs[1, 0].annotate(r'$\rho_s (\Delta_{200c}(a_i), m(a_j))$', xy=(0.03, 0.93), xycoords='axes fraction',
                   fontsize=16)
# Get handle to the image from the last plot
im, boundaries = plots.plot_2dcorr(fm2200c, scales=scales, xlabel=r'$a_i$',
                                   ylabel=r'$a_j$',
                                   ax=axs[1, 1])
axs[1, 1].annotate(r'$\rho_s (f_{m2, 200c}(a_i), m(a_j))$', xy=(0.03, 0.93), xycoords='axes fraction',
                   fontsize=16)

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
    sec.set_xticks(lb_ticks)
    sec.set_xticklabels([f"{t:.1f}" for t in lb_ticks])

    sec.set_xlabel("Lookback Time (Gyr)", size=13)

cbar = fig.colorbar(im, ax=axs.ravel().tolist(),
                    boundaries=boundaries, ticks=boundaries,
                    location='right', fraction=0.05)

cbar.ax.set_yticklabels([f'{b:.1f}' for b in boundaries])
cbar.set_label(
    r'Correlation coefficient  $\rho_s (DS_{200c}(a_i), m(a_j))$', fontsize=20)

plt.tight_layout(rect=[0, 0, 0.84, 1])  # leave space on right for colorbar
# plt.savefig('plots/ds_200c_ma.pdf')
plt.show()

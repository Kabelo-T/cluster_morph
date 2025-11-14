import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.colors import LogNorm
import seaborn as sns

import utils.file as futils
import utils.data as datutils


plt.rcParams['xtick.labelsize'] = 13
plt.rcParams['ytick.labelsize'] = 13
plt.rcParams['axes.labelsize'] = 18
plt.rcParams["axes.titlesize"] = 18
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["font.family"] = "serif"

# Colorblind-friendly color palette https://www.nature.com/articles/nmeth.1618.pdf
colors = [
    '#000000',      # Black
    '#0072B2',      # Blue
    '#F0E442',      # Yellow
    '#009E73',      # Bluish Green
    '#CC79A7',      # Reddish Purple
    '#D55E00',      # Vermilion
    '#56B4E9',      # Sky Blue
    '#E69F00'       # Orange
]

plt.rc('axes', prop_cycle=plt.cycler('color', colors))


def display_img(image: np.array, axs, segmap: np.array = None,
                mask: np.array = None, vmin=None, vmax=None, **kwargs):
    """Display an image, masked or unmasked. May plot alongside segmentation map

    Parameters
    ----------
    image : np.array
        the original image
    segmap : np.array, optional
        segmentation map that labels distinct sources in image, by default None
    mask : np.array[bool], optional
        a mask for cropping the image, by default None
    """
    if mask is not None:
        image = np.where(mask, image, 0)

    if vmin is not None and vmax is not None:
        norm = LogNorm(vmin=vmin, vmax=vmax)
    else:
        norm = LogNorm()

    if isinstance(axs, np.ndarray):
        axs[0].imshow(image, norm=norm)
        axs[1].imshow(segmap, norm=norm, cmap='gray')
    else:
        axs.imshow(image, norm=norm)
    return


def plot_light(object, zoomed_region, zoom_size):
    """
    For a given patch of the image plot the region and overplot half-light radius r50, r80 and r20
    """
    fig, ax = plt.subplots()
    ax.imshow(zoomed_region, cmap='gray', origin='lower', vmin=0, vmax=0.05)

    circle_half = Circle((zoom_size, zoom_size), object.rhalf_circ,
                         color='red', fill=False, linewidth=1.5, label='r_half')
    circle_80 = Circle((zoom_size, zoom_size), object.r80,
                       color='white', fill=False, linewidth=1.5, label='r_80')
    circle_20 = Circle((zoom_size, zoom_size), object.r20,
                       color='yellow', fill=False, linewidth=1.5, label='r_20')

    ax.add_patch(circle_half)
    ax.add_patch(circle_80)
    ax.add_patch(circle_20)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.legend()
    plt.show()

    return


def plot_corr_matrix(corr_matrix: pd.DataFrame, title: str, save: bool = False,
                     fname: str = None, show_cbar: bool = False,
                     annot_size: int = 20, **kwargs):
    """Plot the Correlation Matrix as a heatmap

    Parameters
    ----------
    corr_matrix : pd.DataFrame
        correlations of all the parameters
    """
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    plt.figure(figsize=(14, 14))
    fhm = sns.heatmap(data=corr_matrix, mask=mask, cmap='coolwarm',
                      square=True, cbar=show_cbar, cbar_kws={"shrink": 0.3},
                      annot_kws={"size": annot_size, "weight": "bold"}, fmt='.2f',  **kwargs)
    plt.title(title)
    plt.tight_layout()
    plt.xticks(fontsize=14, rotation=45, ha='right')
    plt.yticks(fontsize=14)
    if save:
        plt.savefig(f'plots/{fname}')
    plt.show()
    return


def high_corr_cols(corr_matrix: pd.DataFrame, tolerance=0.7) -> list[bool]:
    """Select columns that are highly correlated and anti-correlated

    Parameters
    ----------
    corr_matrix : pd.DataFrame
        correlations of all the parameters

    Returns
    -------
    list[bool]
        _description_
    """
    selection = []
    good_cols = []
    for i, x in enumerate(corr_matrix):
        lst = corr_matrix[x][np.abs(corr_matrix[x]) > 0.8]
        good_cols.append(x)
        for c in lst.index:
            if c in good_cols:
                continue
            else:
                selection.append((x, c))
    return selection


def plot_corr(corr_matrix: pd.DataFrame, morph_df: pd.DataFrame,
              low: int = 30, high=40) -> None:
    """Plot joint distributions of the highly correlated and anti-correlated 
    parameters

    Parameters
    ----------
    corr_matrix : pd.DataFrame
        correlations of all the parameters
    morph_df : pd.DataFrame
        statmorph morphological measurements
    low : int
        lower bound index
    high : int
        upper bound index
    """

    selection = high_corr_cols(corr_matrix=corr_matrix)
    for i, row in enumerate(selection):
        if i in range(30, 40):
            rho = corr_matrix[row[0]][row[1]]
            df = morph_df[[row[0], row[1]]]
            joint = sns.jointplot(data=df, x=row[0], y=row[1], height=4)
            joint.figure.suptitle("r = " + str(rho))
            plt.tight_layout()
            plt.subplots_adjust(top=0.9)
            plt.show()


def plot_mah(indx: int, axs: plt.Axes, am=False, highlight=False,
             mah_dir: str = 'data/gadgetx3k/AHFHaloHistory/',
             show_relaxed: bool = False, show_disturbed: bool = False) -> None:

    state = {0: "Disturbed",
             1: "Relaxed"}

    dsdf = futils.get_ds(snap=125, clean=False)
    mah_df_dict = futils.get_mah_all(mah_dir=mah_dir)
    ds = int(dsdf.loc[[indx]]['DS_200[2]'].values[0])
    axs.set_title(f'ID {indx} | {state[ds]}')

    if am:
        axs.set_xlabel('m = M/M(z=0)')
        axs.set_ylabel('a(m)')
        for k, df in mah_df_dict.items():
            if k == indx:
                axs.plot(df['M/M0'], df['aexp'], lw=5, color='red')
            else:
                ds = int(dsdf.loc[[k]]['DS_200[2]'].values[0])
                if (ds == 1 and show_relaxed) or (ds == 0 and show_disturbed):
                    axs.plot(df['M/M0'], df['aexp'],
                             color='black', alpha=0.3, lw=1)

    else:
        axs.set_xlabel('Scale Factor a = 1/(1+z)')
        axs.set_ylabel('m(a)')
        for k, df in mah_df_dict.items():
            if k == indx:
                axs.plot(df['aexp'], df['M/M0'], lw=5, color='red')
            else:
                ds = int(dsdf.loc[[k]]['DS_200[2]'].values[0])
                if (ds == 1 and show_relaxed) or (ds == 0 and show_disturbed):
                    axs.plot(df['aexp'], df['M/M0'],
                             color='black', alpha=0.3, lw=1)

    return


def plot_ma_corrs(data_dict: dict, params: list[str], labels: list[str], aexp: np.array,
                  axs, sm=True) -> None:

    yticks = np.arange(0, 0.9, 0.1)
    axs.set_xlabel('Scale Factor a = 1/(1+z)')

    if sm:
        axs.set_ylabel(r'$|\rho_s (m(a)_{SM}, m(a))|$')
        ylim = (0., 0.7)
    else:
        axs.set_ylabel(r'$|\rho_s (DS_{a=1}, m(a))|$')
        ylim = (0., 0.8)

    for i, param in enumerate(params):
        p50 = datutils.get_perc(
            data_dict, param=param, q=50, history='M/M0')
        p50 = np.abs(p50)
        axs.plot(aexp, p50, color=colors[i], label=labels[i])

        if param == 'A' or param == 'core_C' or param == 'fm_200[5]' or param == 'eta_200[3]':
            p25 = datutils.get_perc(
                data_dict, param=param, q=25, history='M/M0')
            p75 = datutils.get_perc(
                data_dict, param=param, q=75, history='M/M0')

            if param == 'A':
                p25[3:] = np.abs(p25[3:])       # hack to fix tail of A corrs
                p75 = np.abs(p75)
            else:
                p25, p75 = np.abs(p25), np.abs(p75)

            axs.fill_between(aexp, p25, p75, color=colors[i], alpha=0.2)
            axs.fill_between(aexp, p25, p75, color=colors[i], alpha=0.2)

    axs.set_yticks(yticks)
    axs.set_ylim(ylim)
    axs.grid()
    axs.legend()

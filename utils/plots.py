import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.colors import LogNorm
import seaborn as sns


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


def plot_corr_matrix(corr_matrix: pd.DataFrame):
    """Plot the Correlation Matrix as a heatmap

    Parameters
    ----------
    corr_matrix : pd.DataFrame
        correlations of all the parameters
    """
    mask = np.triu(np.ones_like(corr_matrix, dtype=np.bool))
    plt.figure(figsize=(25, 25))
    fhm = sns.heatmap(corr_matrix, mask=mask,
                      cmap='coolwarm', square=True)
    plt.title('Full Spearman Correlation Matrix Heatmap')
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

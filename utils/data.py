import numpy as np
import pandas as pd
from photutils.segmentation import detect_threshold, detect_sources
from skimage import draw


def segmentation_map(image, nsigma=5.5, npixels=10000, mask=None):
    """
    Create the segmentation map for use in statmorph.

    Parameters
    ----------
    image: input image data
    nsigma: The number of standard deviations per pixel above the ``background``
            for which to consider a pixel as possibly being
            part of a source.
    npixels: The minimum number of connected pixels an object must have to be detected.
    mask: A boolean mask where `True` values indicate masked pixels

    Returns
    -------

    segmap: 2D segmentation map for statmorph
    """
    threshold = detect_threshold(image, nsigma)
    if mask is None:
        segmap = detect_sources(image, threshold, npixels=npixels)
    else:
        segmap = detect_sources(image, threshold, npixels=npixels, mask=mask)
    return segmap


def crop(image, radius=1000):
    """
    Square image crop.
    """
    w = image.shape[1]
    h = image.shape[0]
    origin = (int(w/2), int(h/2))
    x_min = max(origin[0] - radius, 0)
    x_max = min(origin[0] + radius, w)
    y_min = max(origin[1] - radius, 0)
    y_max = min(origin[1] + radius, h)
    cropped_img = image[y_min:y_max, x_min:x_max]
    return cropped_img


def zoom_in(object, image):
    """
    Square image crop using statmorph source object properties.
    """
    zoom_size = 4 * object.rhalf_circ
    x_min = int(object.xc_centroid - zoom_size)
    x_max = int(object.xc_centroid + zoom_size)
    y_min = int(object.yc_centroid - zoom_size)
    y_max = int(object.yc_centroid + zoom_size)

    # Extract the zoomed region
    zoomed_region = image[y_min:y_max, x_min:x_max]
    return zoomed_region, zoom_size


def circular_mask(image, image_center, radius):
    segmap = np.zeros(image.shape)
    rr, cc = draw.disk((image_center[1], image_center[0]),
                       radius=radius,
                       shape=(image.shape[0], image.shape[1]))
    segmap[rr, cc] = 1
    return segmap == 1


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
    for src in source_morphs:
        sources.append({
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
            # something is up with these double sersic fits that makes this run forever
            # 'doublesersic_aic': src.doublesersic_aic,
            # 'doublesersic_amplitude1': src.doublesersic_amplitude1,
            # 'doublesersic_amplitude2': src.doublesersic_amplitude2,
            # 'doublesersic_bic': src.doublesersic_bic,
            # 'doublesersic_chi2_dof': src.doublesersic_chi2_dof,
            # 'doublesersic_ellip1': src.doublesersic_ellip1,
            # 'doublesersic_ellip2': src.doublesersic_ellip2,
            # 'doublesersic_n1': src.doublesersic_n1,
            # 'doublesersic_n2': src.doublesersic_n2,
            # 'doublesersic_rhalf1': src.doublesersic_rhalf1,
            # 'doublesersic_rhalf2': src.doublesersic_rhalf2,
            # 'doublesersic_theta1': src.doublesersic_theta1,
            # 'doublesersic_theta2': src.doublesersic_theta2,
            # 'doublesersic_xc': src.doublesersic_xc,
            # 'doublesersic_yc': src.doublesersic_yc,
            'sky_mean': src.sky_mean,
            'sky_median': src.sky_median,
            'sky_sigma': src.sky_sigma,
            'flag': src.flag,
            'flag_sersic': src.flag_sersic,
            'flux_circ': src.flux_circ,
            'flux_ellip': src.flux_ellip,
            'runtime': src.runtime
        })
    sources = pd.DataFrame(sources)
    if save:
        if name is not None:
            sources.to_csv(name)
        else:
            sources.to_csv('source_morphs.csv')
    return sources

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import seaborn as sns
from skimage import draw
from photutils.segmentation import detect_threshold, detect_sources

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

def display_img(image, segmap=None):
    """
    Display the image. Can plot alongside segmap.
    """
    if segmap is not None:
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10,10))
        axs[0].imshow(image, vmin=0, vmax=0.05)
        axs[1].imshow(segmap, cmap='gray')
    else:
        fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(6,6))
        axs.imshow(image, vmin=0, vmax=0.05) 

    plt.show()
    return

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

def plot_light(object, zoomed_region, zoom_size):
    """
    For a given patch of the image plot the region and overplot half-light radius r50, r80 and r20
    """
    #zoomed_region = zoom_in(object)
    fig, ax = plt.subplots()
    ax.imshow(zoomed_region, cmap='gray', origin='lower', vmin=0, vmax=0.05)

    # Plot the half-light radius circle
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

def create_mask(image, center, radius):
    w = image.shape[1]
    h = image.shape[0]
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    # masked_img = np.zeros_like(drz)
    # masked_img[mask] = drz[mask]
    return mask

def circular_segmentation_map(image, image_center, radius):
    segmap = np.zeros(image.shape)
    rr, cc = draw.disk((image_center[1], image_center[0]),
                        radius=radius, 
                        shape=(image.shape[0], image.shape[1]))
    segmap[rr, cc] = 1
    return segmap == 1

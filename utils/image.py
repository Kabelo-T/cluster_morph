import numpy as np
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


def circular_segmap(image: np.ndarray, image_center: tuple[int, int],
                    radius: int) -> np.ndarray[int]:
    """Create a circular segmentation map, all 1s within radius and 0s outside.

    Parameters
    ----------
    image : np.ndarray
    image_center : tuple[int, int]
        image center in pixel coordinates
    radius : int
        radius in pixels

    Returns
    -------
    segmap : np.ndarray[int]
    """

    segmap = np.zeros(image.shape)
    rr, cc = draw.disk((image_center[1], image_center[0]),
                       radius=radius,
                       shape=(image.shape[0], image.shape[1]))
    segmap[rr, cc] = 1
    return segmap.astype(np.uint8)


def annular_mask(image: np.ndarray, image_center: tuple[int, int], r1: float,
                 r2: float) -> np.ndarray[int]:
    """Creates an annular mask for statmorph measurement

    Parameters
    ----------
    image : np.ndarray
    image_center : tuple[int, int]
        image center in pixel coordinates
    r1 : float
        Inner radius in physical units
    r2 : float
        Outer radius in physical units

    Returns
    -------
    segmap : np.ndarray[int]
    """

    segmap = circular_segmap(image, image_center, radius=r2)
    rr, cc = draw.disk((image_center[1], image_center[0]),
                       radius=r1,
                       shape=(image.shape[0], image.shape[1]))
    segmap[rr, cc] = 0
    return segmap.astype(np.uint8)

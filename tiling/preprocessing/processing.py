"""Furcifar Processing Module

This module contains functions to perform a preprocessing of whole-slide images of the
CAMELYON data sets to be used  to train a convolutional neural network for metastasis
localisation.
"""

import logging
import math
from typing import Tuple, Iterator
from datetime import datetime

import numpy as np
from skimage import filters
from skimage.draw import polygon as ski_polygon
from skimage.measure import label as ski_label

from .datamodel import Slide
from .util import ProgressBar, LogStyleAdapter, load_color_normalization_values
from .histopath import filter_grays, filter_remove_small_holes, filter_remove_small_objects

logger = LogStyleAdapter(logging.getLogger('preprocessing.processing'))


def apply_image_filters(np_img, remove_object_size=5000, remove_holes_size=3000):
    """
    Apply filters to image as NumPy array
    Args:
    np_img: Image as a NumPy array of type bool.
    remove_object_size: Remove small objects below this size.
    remove_holes_size: Remove small holes below this size.
    output_type: Type of array to return (bool, float, or uint8).

    Returns:
    NumPy array (bool, float, or uint8).
    """
    # in case there is an alpha channel in the image (X, X, 4)
    np_img = remove_alpha_channel(np_img)
    mask_not_grays = filter_grays(np_img)
    mask_remove_objects = filter_remove_small_objects(mask_not_grays, min_size=remove_object_size,
                                                      output_type="bool")
    mask_remove_holes = filter_remove_small_holes(mask_remove_objects, min_size=remove_holes_size,
                                                  output_type="bool")
    return mask_remove_holes


def remove_alpha_channel(image: np.ndarray) -> np.ndarray:
    """Remove the alpha channel of an image.

    Parameters
    ----------
    image : np.ndarray
        RGBA image as numpy array with W×H×C dimensions.


    Returns
    -------
    np.ndarray
        RGB image as numpy array
    """
    if len(image.shape) == 3 and image.shape[2] == 4:
        return image[::, ::, 0:3:]
    else:
        return image


def rgb2gray(rgb: np.ndarray) -> np.ndarray:
    """Convert RGB color image to a custom gray scale for HE-stained WSI

    by Jonas Annuscheit


    Parameters
    ----------
    rgb : np.ndarray
        Color image.


    Returns
    -------
    np.ndarray
        Gray scale image as float64 array.
    """
    gray = 1.0 * rgb[::, ::, 0] + rgb[::, ::, 2] - (
        (1.0 * rgb[::, ::, 0] + rgb[::, ::, 1] + rgb[::, ::, 2])
        / 1.5)
    gray[gray < 0] = 0
    gray[gray > 255] = 255
    return gray


def create_otsu_mask_by_threshold(image: np.ndarray, threshold) -> np.ndarray:
    """Create a binary mask separating fore and background based on the otsu threshold.

    Parameters
    ----------
    image : np.ndarray
        Gray scale image as array W×H dimensions.

    threshold : float
        Upper Otsu threshold value.


    Returns
    -------
    np.ndarray
        The generated binary masks has value 1 in foreground areas and 0s everywhere
        else (background)
    """
    otsu_mask = image > threshold
    otsu_mask2 = image > threshold * 0.25

    otsu_mask2_labeled = ski_label(otsu_mask2)
    for i in range(1, otsu_mask2_labeled.max()):
        if otsu_mask[otsu_mask2_labeled == i].sum() == 0:
            otsu_mask2_labeled[otsu_mask2_labeled == i] = 0
    otsu_mask3 = otsu_mask2_labeled
    otsu_mask3[otsu_mask3 > 0] = 1

    return otsu_mask3.astype(np.uint8)


def get_otsu_threshold(slide: Slide, level=3) -> float:
    """Calculate the otsu threshold of a slide on layer `level`.


    Parameters
    ----------
    slide: Slide
        Whole image slide
    level: int
        Whole image slide layer number


    Returns
    -------
    otsu_threshold: float
        Upper threshold value. All pixels with an intensity higher than
        this value are assumed to be foreground.


    References
    ----------
    Wikipedia, http://en.wikipedia.org/wiki/Otsu's_Method


    See Also
    --------
    skimage.filters.threshold_otsu
    """
    if level < 3:
        logger.warning('Level under 3 might cause memory overflows!')

    img = remove_alpha_channel(np.asarray(slide.get_full_slide(level)))
    gray_slide = rgb2gray(img)
    return filters.threshold_otsu(gray_slide, nbins=256)


def split_negative_slide(slide: Slide, level, otsu_threshold,
                         poi_threshold=0.2, tile_size=256,
                         overlap=5, verbose: bool = False, use_upstream_filters=False):
    """Create tiles from a negative slide.

    Iterator over the slide in `tile_size`×`tile_size` Tiles. For every tile an otsu mask
    is created and summed up. Only tiles with sums over the percentage threshold
    `poi_threshold` will be yield.

    Parameters
    ----------
    slide : Slide
        Input Slide.

    level : int
        Layer to produce tiles from.

    otsu_threshold : float
        Otsu threshold of the whole slide on layer `level`.

    poi_threshold : float, optional
        Minimum percentage, 0 to 1, of pixels with tissue per tile. (Default 0.01; 1%)

    tile_size : int
        Pixel size of one side of a square tile the image will be split into.
        (Default: 128)

    overlap : int, optional
        Count of pixel overlapping between two tiles. (Default: 30)

    use_upstream_filters: bool
        clean the slide by applying extra filters and generating a more precise mask
        for now we keep it to False per default as it is very memory intensive

    verbose : Boolean, optional
        If set to True a progress bar will be printed to stdout. (Default: False)


    Yields
    -------
    image_tile : np.ndarray
        Array of (`tile_size`, `tile_size`) shape containing tumorous tissue.

    bounds : tuple
        Tile boundaries on layer 0: ((x, y), (width, height))
    """
    if tile_size <= overlap:
        raise ValueError("Overlap has to be smaller than the tile size.")
    if overlap < 0:
        raise ValueError("Overlap can not be negative.")
    if otsu_threshold < 0:
        raise ValueError("Otsu threshold can not be negative.")
    if not 0.0 <= poi_threshold <= 1.0:
        raise ValueError("PoI threshold has to be between 0 and 1")

    width0, height0 = slide.level_dimensions[0]
    downsample = slide.level_downsamples[level]

    if use_upstream_filters:

        mask_filters = apply_image_filters(np.asarray(slide.get_full_slide(level=level)),
                                           remove_object_size=5000, remove_holes_size=3000)

    # tile size on level 0
    tile_size0 = int(tile_size * downsample + 0.5)
    overlap0 = int(overlap * downsample + 0.5)

    if verbose:
        # amount of tiles horizontally and vertically
        count_horizontal = int(
            math.ceil((width0 - (tile_size0 - overlap0)) / (tile_size0 - overlap0) + 1))
        count_vertical = int(
            math.ceil((height0 - (tile_size0 - overlap0)) / (tile_size0 - overlap0) + 1))

        bar_suffix = '%(percent)3d%% | Tiles: %(index)d / %(max)d ' \
                     '[%(elapsed_fmt)s | eta: %(remaining_fmt)s]'
        bar = ProgressBar(f'Processing: {slide.name:20}',
                          max=count_horizontal * count_vertical,
                          suffix=bar_suffix)
        print('verbose on')
        print('count_vertical:', count_vertical)
        print('count_horitonzal:', count_horizontal)

    min_poi_count = tile_size ** 2 * poi_threshold

    for yi, y in enumerate(range(0, height0, tile_size0 - overlap0)):
        if verbose:
            print('row', yi, 'of', count_vertical, ' -- time: ', datetime.now())
        for xi, x in enumerate(range(0, width0, tile_size0 - overlap0)):
            tile = np.asarray(slide.read_region((x, y), level, (tile_size, tile_size)))
            mask = create_otsu_mask_by_threshold(rgb2gray(tile), otsu_threshold)

            if use_upstream_filters:
                x_reduced, y_reduced = int(x/downsample), int(y/downsample)
                mask_f = mask_filters[y_reduced:y_reduced+tile_size, x_reduced:x_reduced+tile_size]
                if mask_f.shape[0] == mask_f.shape[1] and mask_f.shape[0] == mask.shape[0]:
                    mask = np.logical_and(mask, mask_f)

            poi_count = np.sum(mask)
            if poi_count >= min_poi_count:
                yield remove_alpha_channel(tile), ((x, y), (tile_size0, tile_size0))
            if verbose:
                bar.next()
    if verbose:
        bar.finish()


def split_test_slide(slide: Slide, level, otsu_threshold,
                     poi_threshold=0.2, tile_size=128,
                     overlap=0, verbose: bool = False, use_upstream_filters=True):
    """Create tiles from a negative slide.

    Iterator over the slide in `tile_size`×`tile_size` Tiles. For every tile an otsu mask
    is created and summed up. Only tiles with sums over the percental threshold
    `poi_threshold` will be yield.

    Parameters
    ----------
    slide : Slide
        Input Slide.

    level : int
        Layer to produce tiles from.

    otsu_threshold : float
        Otsu threshold of the whole slide on layer `level`.

    poi_threshold : float, optional
        Minimum percentage, 0 to 1, of pixels with tissue per tile. (Default 0.01; 1%)

    tile_size : int
        Pixel size of one side of a square tile the image will be split into.
        (Default: 128)

    overlap : int, optional
        Count of pixel overlapping between two tiles. (Default: 30)

    use_upstream_filters: bool
        clean the slide by applying extra filters and generating a more precise mask
        for now we keep it to False per default as it is very memory intensive

    verbose : Boolean, optional
        If set to True a progress bar will be printed to stdout. (Default: False)


    Yields
    -------
    image_tile : np.ndarray
        Array of (`tile_size`, `tile_size`) shape containing tumorous tissue.

    bounds : tuple
        Tile boundaries on layer 0: ((x, y), (width, height))
    """
    if tile_size <= overlap:
        raise ValueError("Overlap has to be smaller than the tile size.")
    if overlap < 0:
        raise ValueError("Overlap can not be negative.")
    if otsu_threshold < 0:
        raise ValueError("Otsu threshold can not be negative.")
    if not 0.0 <= poi_threshold <= 1.0:
        raise ValueError("PoI threshold has to be between 0 and 1")

    if use_upstream_filters:
        mask_filters = apply_image_filters(np.asarray(slide.get_full_slide(level=level)),
                                           remove_object_size=5000, remove_holes_size=3000)

    width0, height0 = slide.level_dimensions[0]
    downsample = slide.level_downsamples[level]

    # tile size on level 0
    tile_size0 = int(tile_size * downsample + 0.5)
    overlap0 = int(overlap * downsample + 0.5)

    if verbose:
        # amount of tiles horizontally and vertically
        count_horizontal = int(
            math.ceil((width0 - (tile_size0 - overlap0)) / (tile_size0 - overlap0) + 1))
        count_vertical = int(
            math.ceil((height0 - (tile_size0 - overlap0)) / (tile_size0 - overlap0) + 1))

        bar_suffix = '%(percent)3d%% | Tiles: %(index)d / %(max)d ' \
                     '[%(elapsed_fmt)s | eta: %(remaining_fmt)s]'
        bar = ProgressBar(f'Processing: {slide.name:20}',
                          max=count_horizontal * count_vertical,
                          suffix=bar_suffix)
        print('verbose on')
        print('count_vertical:', count_vertical)
        print('count_horitonzal:', count_horizontal)

    min_poi_count = tile_size ** 2 * poi_threshold if poi_threshold is not None else 1

    for yi, y in enumerate(range(0, height0, tile_size0 - overlap0)):
        if verbose:
            print('row', yi, 'of', count_vertical, ' -- time: ', datetime.now())
        for xi, x in enumerate(range(0, width0, tile_size0 - overlap0)):
            tile = np.asarray(slide.read_region((x, y), level, (tile_size, tile_size)))
            # if slide has annotations it is a tumor slide
            # if tumor mask exists, it's a tumor tile.
            mask = create_tumor_mask(slide, level, ((x, y), (tile_size, tile_size)))
            poi_count = np.sum(mask)
            if poi_count > 1:
                # return tile np array, coordinates and label = 1
                yield remove_alpha_channel(tile), ((x, y), (tile_size0, tile_size0)), 1
            # otherwise it could be empty or regular tissue
            else:
                mask = create_otsu_mask_by_threshold(rgb2gray(tile), otsu_threshold)

                if use_upstream_filters:
                    x_reduced, y_reduced = int(x / downsample), int(y / downsample)
                    mask_f = mask_filters[y_reduced:y_reduced + tile_size,
                                          x_reduced:x_reduced + tile_size]
                    if mask_f.shape[0] == mask_f.shape[1] and mask_f.shape[0] == mask.shape[0]:
                        mask = np.logical_and(mask, mask_f)

                poi_tissue_count = np.sum(mask)
                # if tissue, check that it is above the poi_threshold and label = 0
                if poi_tissue_count >= min_poi_count:
                    yield remove_alpha_channel(tile), ((x, y), (tile_size0, tile_size0)), 0
            if verbose:
                bar.next()
    if verbose:
        bar.finish()


def create_tumor_mask(slide: Slide, level, bounds=None):
    """Create a tumor mask for a slide or slide section.

    If `bounds` is given the tumor mask of only the section of the slide will be
    calculated.


    Parameters
    ----------
    slide : Slide
        Tissue slide.

    level : int
        Slide layer.

    bounds : tuple, optional
        Boundaries of a section as: ((x, y), (width, height))
        Where x and y are coordinates of the top left corner of the slide section on
        layer 0 and width and height the dimensions of the section on the specific
        layer `level`.  (Default: None)


    Returns
    -------
    tumor_mask : np.ndarray
        Binary tumor mask of the specified section. Healthy tissue is represented by 0,
        cancerous by 1.
    """
    if bounds is None:
        start_pos = (0, 0)
        size = slide.level_dimensions[level]
    else:
        start_pos, size = bounds

    mask = np.zeros((size[1], size[0]), dtype=np.uint8)
    downsample = slide.level_downsamples[level]

    for i, annotation in enumerate(slide.annotations):
        c_values, r_values = list(zip(*annotation.polygon))
        r = np.array(r_values, dtype=np.float32)
        r -= start_pos[1]
        r /= downsample
        r = np.array(r + 0.5, dtype=np.int32)

        c = np.array(c_values, dtype=np.float32)
        c -= start_pos[0]
        c /= downsample
        c = np.array(c + 0.5, dtype=np.int32)

        rr, cc = ski_polygon(r, c, shape=mask.shape)
        mask[rr, cc] = 1

    return mask


def split_positive_slide(slide: Slide, level, tile_size=128, overlap=5,
                         poi_threshold=None,
                         verbose=False) -> Iterator[Tuple[np.ndarray, Tuple]]:
    """Create tiles from a positive slide.

    Iterator over the slide in `tile_size`×`tile_size` Tiles. For every tile a tumor mask
    is created and summed up.

    Parameters
    ----------
    slide : Slide
        Input Slide.

    level : int
        Layer to produce tiles from.

    tile_size : int, optional
        Pixel size of one side of a square tile the image will be split into.
        (Default: 128)

    overlap : int, optional
        Count of pixel overlapping between two tiles. (Default: 30)

    poi_threshold : float or None, optional
        Minimum percentage, 0 to 1, of pixels with metastasis per tile or None for tiles
        with at least one tumor pixel. (Default: None)

    verbose : Boolean, optional
        If set to True a progress bar will be printed to stdout. (Default: False)


    Yields
    -------
    image_tile : np.ndarray
        Array of (`tile_size`, `tile_size`) shape containing tumorous tissue.

    bounds : tuple
        Tile boundaries on layer 0: ((x, y), (width, height))

    verbose : Boolean, optional
        If set to True a progress bar will be printed to stdout. (Default: False)
    """
    if not slide.annotations:
        raise ValueError("Slide {} has no annotations.".format(slide.name))
    if tile_size <= overlap:
        raise ValueError("Overlap has to be smaller than tile_size.")
    if overlap < 0:
        raise ValueError("Overlap can not be negative.")

    width0, height0 = slide.level_dimensions[0]
    downsample = slide.level_downsamples[level]

    # tile size on level 0
    tile_size0 = int(tile_size * downsample + 0.5)
    overlap0 = int(overlap * downsample + 0.5)

    if verbose:
        count_horitonzal = int(
            math.ceil((width0 - (tile_size0 - overlap0)) / (tile_size0 - overlap0) + 1))
        count_vertical = int(
            math.ceil((height0 - (tile_size0 - overlap0)) / (tile_size0 - overlap0) + 1))

        bar_suffix = '%(percent)3d%% | Tiles: %(index)d / %(max)d ' \
                     '[%(elapsed_fmt)s | eta: %(remaining_fmt)s]'
        bar = ProgressBar(f'Processing: {slide.name:20}',
                          max=count_horitonzal * count_vertical,
                          suffix=bar_suffix)
    min_poi_count = tile_size ** 2 * poi_threshold if poi_threshold is not None else 1
    for yi, y in enumerate(range(0, height0, tile_size0 - overlap0)):
        mask_row = create_tumor_mask(slide, level, ((0, y), (width0, tile_size)))
        if mask_row.sum() > 0: ### lets skip rows without any tumor
            for xi, x in enumerate(range(0, width0, tile_size0 - overlap0)):
                
                if level != 0:
                    mask = create_tumor_mask(slide, level, ((x, y), (tile_size, tile_size)))
                    poi_count = np.sum(mask)
                if level == 0:
                    poi_count = np.sum(mask_row[:, x:(x+tile_size)])
                    
                logger.debug('Tile ({:2},{:2}) PoI count: {:6,}', yi, xi, poi_count)
                if poi_count >= min_poi_count:
                    tile = slide.read_region((x, y), level, (tile_size, tile_size))

                    tile = remove_alpha_channel(np.asarray(tile))
                    yield tile, ((x, y), (tile_size0, tile_size0))

                if verbose:
                    bar.next()

    if verbose:
        bar.finish()


def split_whole_slide(slide: Slide, level, tile_size=256,
                      color_normalization_file="CAMELYON16_color_normalization.json",
                      green_layer_only=False, verbose=False) -> Iterator[Tuple[np.ndarray, Tuple]]:
    """Create tiles from a positive slide.

    Iterator over the slide in `tile_size`×`tile_size` Tiles. For every tile a tumor mask
    is created and summed up.

    Parameters
    ----------
    slide : Slide
        Input Slide.

    level : int
        Layer to produce tiles from.

    tile_size : int, optional
        Pixel size of one side of a square tile the image will be split into.
        (Default: 128)

    verbose : Boolean, optional
        If set to True a progress bar will be printed to stdout. (Default: False)


    Yields
    -------
    image_tile : np.ndarray
        Array of (`tile_size`, `tile_size`) shape containing tumorous tissue.

    bounds : tuple
        Tile boundaries on layer 0: ((x, y), (width, height))

    verbose : Boolean, optional
        If set to True a progress bar will be printed to stdout. (Default: False)
    """

    width0, height0 = slide.level_dimensions[0]
    downsample = slide.level_downsamples[level]

    mean, std = load_color_normalization_values(color_normalization_file)
    # tile size on level 0
    tile_size0 = int(tile_size * downsample + 0.5)

    if verbose:
        count_horitonzal = int(math.ceil((width0 - tile_size0) / tile_size0 + 1))
        count_vertical = int(math.ceil((height0 - tile_size0) / tile_size0 + 1))

        bar_suffix = '%(percent)3d%% | Tiles: %(index)d / %(max)d ' \
                     '[%(elapsed_fmt)s | eta: %(remaining_fmt)s]'
        bar = ProgressBar(f'Processing: {slide.name:20}',
                          max=count_horitonzal * count_vertical,
                          suffix=bar_suffix)

    for yi, y in enumerate(range(0, height0, tile_size0)):
        for xi, x in enumerate(range(0, width0, tile_size0)):
            label = 0

            if level == 0:
                poi_count = np.sum(mask_row[:, x:(x + tile_size)])
            else:
                mask = create_tumor_mask(slide, level, ((x, y), (tile_size, tile_size)))
                poi_count = np.sum(mask)
            if verbose:
                print('Tile ({:2},{:2}) PoI count: {:6,}'.format(yi, xi, poi_count))

            if poi_count >= 0:
                label = 1

            tile = slide.read_region((x, y), level, (tile_size, tile_size))
            tile = remove_alpha_channel(np.asarray(tile))
            ntile = tile / 255.0

            for i in [0, 1, 2]:
                ntile[:, :, i] = (ntile[:, :, i] - mean[i]) / std[i]

            if green_layer_only:
                ntile = np.dot(ntile[..., :3], [0.0, 1.0, 0.0])

            # tiles might be shorted on edges, need to resize for segmentation model
            if ntile.shape != (tile_size, tile_size, 3):
                ntile = ntile.resize((tile_size, tile_size, 3))

            yield ntile, label, (xi, yi), ((x, y), (tile_size, tile_size))

            if verbose:
                bar.next()

    if verbose:
        bar.finish()

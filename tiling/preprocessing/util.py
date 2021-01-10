"""Furcifar Utility Modul

This module contains functions for visualisation, logging and progress output during the
preprocessing of whole-slide images of the CAMELYON data sets.
"""

from collections import namedtuple
from datetime import datetime
import fnmatch
import logging
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont
from progress.bar import IncrementalBar
from typing import Dict


Point = namedtuple('Point', 'x y')
# If True, display additional NumPy array stats (min, max, mean, is_binary).
ADDITIONAL_NP_STATS = False


class LogMessage(object):
    def __init__(self, fmt, args):
        self.fmt = fmt
        self.args = args

    def __str__(self):
        return self.fmt.format(*self.args)


class LogStyleAdapter(logging.LoggerAdapter):
    """Style Adapter to allow Python 3 styled string format with '{}'."""

    def __init__(self, logger, extra=None):
        super(LogStyleAdapter, self).__init__(logger, extra or {})

    def log(self, level, msg, *args, **kwargs):
        if self.isEnabledFor(level):
            msg, kwargs = self.process(msg, kwargs)
            self.logger._log(level, LogMessage(msg, args), (), **kwargs)


def find_files(pattern, path) -> Dict[str, str]:
    """Find files in a directory by given file name pattern.

    Parameters
    ----------
    pattern : str
        File pattern allowing wildcards.

    path : str
        Root directory to search in.


    Returns
    -------
    dict(str: str)
        Dictionary of all found files where the file names are keys and the relative paths
        from search root are values.
    """
    result = {}
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result[name] = os.path.join(root, name)
    return result


class ProgressBar(IncrementalBar):
    @property
    def remaining_fmt(self):
        m, s = divmod(self.eta, 60)
        h, m = divmod(m, 60)
        return f'{h:02}:{m:02}:{s:02}'

    @property
    def elapsed_fmt(self):
        m, s = divmod(self.elapsed, 60)
        h, m = divmod(m, 60)
        return f'{h:02}:{m:02}:{s:02}'


def draw_polygon(image: Image.Image, polygon, *, fill, outline) -> Image.Image:
    """Draw a filled polygon on to an image.

    Parameters
    ----------
    image : Image.Image
        Background image to be drawn on.

    polygon :
        Polygon to be drawn.

    fill : color str or tuple
        Fill color.

    outline : color str or tuple
        Outline color.


    Returns
    -------
    Image.Image
        A copy of the background image with the polygon drawn onto.
    """
    img_back = image
    img_poly = Image.new('RGBA', img_back.size)
    img_draw = ImageDraw.Draw(img_poly)
    img_draw.polygon(polygon, fill, outline)
    img_back.paste(img_poly, mask=img_poly)
    return img_back


def get_relative_polygon(polygon, origin: Point, downsample=1):
    """Translate the polygon to relative to a point.


    Parameters
    ----------
    polygon : Sequence[Point]
        Polygon points.

    origin : Point
        The new origin the polygons points shall be relative to.

    downsample : int, optional
        Layer downsample >= 1 (Default: 1)


    Returns
    -------
    tuple(Point)
        New polygon with points relative to origin.
    """
    rel_polygon = []
    for point in polygon:
        rel_polygon.append(Point((point.x - origin.x) / downsample,
                                 (point.y - origin.y) / downsample))

    return tuple(rel_polygon)


class TileMap:
    """Visualisation for slide tiles.

    Creates an image with with tile boundaries drawn over the slide image visualisation
    purposes.

    Attributes
    ----------
        image : PIL.Image.Image
            Map that displays the slide with each added tile drawn over it.
    """

    def __init__(self, slide: "Slide", level=None, fill=(20, 180, 8, 80),
                 outline=(20, 180, 8)):
        """
        Parameters
        ----------
        slide : Slide
            Tissue slide.

        level
            Slide Layer.

        fill : PIL color, optional
            Tile fill color.

        outline : PIL color, optional
            Tile outline color.
        """
        self._slide = slide
        if level is None:
            self._level = slide.level_count - 1
        else:
            self._level = level

        self._fill = fill
        self._outline = outline
        self._downsample = slide.level_downsamples[self._level]
        self.tiles = []
        self.image = slide.get_full_slide(self._level)

    def __repr__(self):
        return '{}(slide={!r}, level={!r})'.format(
            type(self).__name__,
            self._slide,
            self._level
        )

    def add_tile(self, bounds):
        """Add a tile to the map.

        Parameters
        ----------
        bounds : Tuple
            Tile boundaries as a tuple of ((x, y), (width, height)) in layer 0 pixel.
        """
        self.tiles.append(bounds)
        (x, y), (width, height) = bounds
        poly = (Point(x, y), Point(x + width, y), Point(x + width, y + height),
                Point(x, y + height))
        rel_poly = get_relative_polygon(poly, Point(0, 0),
                                        downsample=self._downsample)

        self.image = draw_polygon(self.image, rel_poly, fill=self._fill,
                                  outline=self._outline)


class Time:
    """
    Class for displaying elapsed time.
    FROM DEEPHISTOPATH
    """

    def __init__(self):
        self.start = datetime.now()

    def elapsed_display(self):
        time_elapsed = self.elapsed()
        print("Time elapsed: " + str(time_elapsed))

    def elapsed(self):
        self.end = datetime.now()
        time_elapsed = self.end - self.start
        return time_elapsed


def np_info(np_arr, name=None, elapsed=None):
    """
    Display information (shape, type, max, min, etc) about a NumPy array.
    FROM DEEPHISTOPATH
    Args:
    np_arr: The NumPy array.
    name: The (optional) name of the array.
    elapsed: The (optional) time elapsed to perform a filtering operation.
    """

    if name is None:
        name = "NumPy Array"
    if elapsed is None:
        elapsed = "---"

    if ADDITIONAL_NP_STATS is False:
        print("%-20s | Time: %-14s  Type: %-7s Shape: %s" % (name, str(elapsed), np_arr.dtype, np_arr.shape))
    else:
        max = np_arr.max()
        min = np_arr.min()
        mean = np_arr.mean()
        is_binary = "T" if (np.unique(np_arr).size == 2) else "F"
        print("%-20s | Time: %-14s Min: %6.2f  Max: %6.2f  Mean: %6.2f  Binary: %s  Type: %-7s Shape: %s" % (
            name, str(elapsed), min, max, mean, is_binary, np_arr.dtype, np_arr.shape))


def pil_to_np_rgb(pil_img):
    """
    Convert a PIL Image to a NumPy array.
    FROM DEEPATHISTO
    Note that RGB PIL (w, h) -> NumPy (h, w, 3).

    Args:
    pil_img: The PIL Image.

    Returns:
    The PIL image converted to a NumPy array.
    """
    t = Time()
    rgb = np.asarray(pil_img)
    np_info(rgb, "RGB", t.elapsed())
    return rgb


def np_to_pil(np_img):
    """
    Convert a NumPy array to a PIL Image.
    FROM DEEPATHISTO
    Args:
    np_img: The image represented as a NumPy array.

    Returns:
     The NumPy array converted to a PIL Image.
    """
    if np_img.dtype == "bool":
        np_img = np_img.astype("uint8") * 255
    elif np_img.dtype == "float64":
        np_img = (np_img * 255).astype("uint8")
    return Image.fromarray(np_img)


def display_img(np_img, text=None, font_path="/Library/Fonts/Arial Bold.ttf", size=48, color=(255, 0, 0),
                background=(255, 255, 255), border=(0, 0, 0), bg=False):
    """
    Convert a NumPy array to a PIL image, add text to the image, and display the image.
    FROM DEEPATHISTO
    Args:
    np_img: Image as a NumPy array.
    text: The text to add to the image.
    font_path: The path to the font to use.
    size: The font size
    color: The font color
    background: The background color
    border: The border color
    bg: If True, add rectangle background behind text
    """
    result = np_to_pil(np_img)
    # if gray, convert to RGB for display
    if result.mode == 'L':
        result = result.convert('RGB')
    draw = ImageDraw.Draw(result)
    if text is not None:
        font = ImageFont.truetype(font_path, size)
        if bg:
            (x, y) = draw.textsize(text, font)
            draw.rectangle([(0, 0), (x + 5, y + 4)], fill=background, outline=border)
        draw.text((2, 0), text, color, font=font)
    result.show()


def mask_rgb(rgb, mask):
    """
    Apply a binary (T/F, 1/0) mask to a 3-channel RGB image and output the result.
    FROM DEEPATHISTO
    Args:
    rgb: RGB image as a NumPy array.
    mask: An image mask to determine which pixels in the original image should be displayed.

    Returns:
    NumPy array representing an RGB image with mask applied.
    """
    t = Time()
    result = rgb * np.dstack([mask, mask, mask])
    np_info(result, "Mask RGB", t.elapsed())
    return result


def mask_percent(np_img):
    """
    Determine the percentage of a NumPy array that is masked (how many of the values are 0 values).
    FROM DEEPHISTOPATH
    Args:
    np_img: Image as a NumPy array.

    Returns:
    The percentage of the NumPy array that is masked.
    """
    if (len(np_img.shape) == 3) and (np_img.shape[2] == 3):
        np_sum = np_img[:, :, 0] + np_img[:, :, 1] + np_img[:, :, 2]
        mask_percentage = 100 - np.count_nonzero(np_sum) / np_sum.size * 100
    else:
        mask_percentage = 100 - np.count_nonzero(np_img) / np_img.size * 100
    return mask_percentage


def tissue_percent(np_img):
    """
    Determine the percentage of a NumPy array that is tissue (not masked).
    FROM DEEPHISTOPATH
    Args:
    np_img: Image as a NumPy array.

    Returns:
    The percentage of the NumPy array that is tissue.
    """
    return 100 - mask_percent(np_img)


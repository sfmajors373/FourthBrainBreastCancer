"""Furcifar Utility Modul

This module contains functions for visualisation, logging and progress output during the
preprocessing of whole-slide images of the CAMELYON data sets.
"""

from collections import namedtuple
import fnmatch
import logging
import os
from PIL import Image, ImageDraw
from progress.bar import IncrementalBar
from typing import Dict


Point = namedtuple('Point', 'x y')


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
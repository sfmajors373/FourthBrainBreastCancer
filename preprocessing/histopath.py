from datetime import datetime
import numpy as np

import skimage.exposure as sk_exposure
from skimage.filters import threshold_otsu
import skimage.filters as sk_filters
from skimage import morphology as sk_morphology
from PIL import Image, ImageDraw, ImageFont

from .processing import rgb2gray

ADDITIONAL_NP_STATS = False

# FROM DEEPHISTOPATH wsi folder


class Time:
    """
    Class for displaying elapsed time.
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


def filter_adaptive_equalization(np_img, nbins=256, clip_limit=0.01, output_type="uint8"):
    """
    Filter image (gray or RGB) using adaptive equalization to increase contrast in image, where contrast in local regions
    is enhanced.
    COMES FROM DEEPHISTOPATH
    Args:
    np_img: Image as a NumPy array (gray or RGB).
    nbins: Number of histogram bins.
    clip_limit: Clipping limit where higher value increases contrast.
    output_type: Type of array to return (float or uint8).

    Returns:
     NumPy array (float or uint8) with contrast enhanced by adaptive equalization.
    """
    t = Time()
    adapt_equ = sk_exposure.equalize_adapthist(np_img, nbins=nbins, clip_limit=clip_limit)
    if output_type == "float":
        pass
    else:
        adapt_equ = (adapt_equ * 255).astype("uint8")
        np_info(adapt_equ, "Adapt Equalization", t.elapsed())
    return adapt_equ


def filter_grays(rgb, tolerance=15, output_type="bool"):
    """
    Create a mask to filter out pixels where the red, green, and blue channel values are similar.
    FROM DEEPHISTOPATH
    Args:
    np_img: RGB image as a NumPy array.
    tolerance: Tolerance value to determine how similar the values must be in order to be filtered out
    output_type: Type of array to return (bool, float, or uint8).

    Returns:
    NumPy array representing a mask where pixels with similar red, green, and blue values have been masked out.
    """
    t = Time()
    rgb = rgb.astype(np.int)
    rg_diff = abs(rgb[:, :, 0] - rgb[:, :, 1]) <= tolerance
    rb_diff = abs(rgb[:, :, 0] - rgb[:, :, 2]) <= tolerance
    gb_diff = abs(rgb[:, :, 1] - rgb[:, :, 2]) <= tolerance
    result = ~(rg_diff & rb_diff & gb_diff)

    if output_type == "bool":
        pass
    elif output_type == "float":
        result = result.astype(float)
    else:
        result = result.astype("uint8") * 255
        np_info(result, "Filter Grays", t.elapsed())
    return result


def filter_remove_small_objects(np_img, min_size=3000, avoid_overmask=True, overmask_thresh=95, output_type="uint8"):
    t = Time()
    rem_sm = np_img.astype(bool)  # make sure mask is boolean
    rem_sm = sk_morphology.remove_small_objects(rem_sm, min_size=min_size)
    mask_percentage = mask_percent(rem_sm)
    if (mask_percentage >= overmask_thresh) and (min_size >= 1) and (avoid_overmask is True):
        new_min_size = int(min_size / 2)
        print("Mask percentage %3.2f%% >= overmask threshold %3.2f%% for Remove Small Objs size %d, so try %d" % (
            mask_percentage, overmask_thresh, min_size, new_min_size))
        rem_sm = filter_remove_small_objects(np_img, min_size=new_min_size, avoid_overmask=avoid_overmask,
                                             overmask_thresh=overmask_thresh, output_type=output_type)
    np_img = rem_sm

    if output_type == "bool":
        pass
    elif output_type == "float":
        np_img = np_img.astype(float)
    else:
        np_img = np_img.astype("uint8") * 255

    np_info(np_img, "Remove Small Objs", t.elapsed())
    return np_img


def filter_remove_small_holes(np_img, min_size=3000, output_type="uint8"):
    """
    Filter image to remove small holes less than a particular size.
    FROM DEEPHISTOPATH
    Args:
    np_img: Image as a NumPy array of type bool.
    min_size: Remove small holes below this size.
    output_type: Type of array to return (bool, float, or uint8).

    Returns:
    NumPy array (bool, float, or uint8).
    """
    t = Time()

    rem_sm = sk_morphology.remove_small_holes(np_img, area_threshold=min_size)

    if output_type == "bool":
        pass
    elif output_type == "float":
        rem_sm = rem_sm.astype(float)
    else:
        rem_sm = rem_sm.astype("uint8") * 255

    np_info(rem_sm, "Remove Small Holes", t.elapsed())
    return rem_sm


def filter_rgb_to_grayscale(np_img, output_type="uint8"):
  """
  Convert an RGB NumPy array to a grayscale NumPy array.
  Shape (h, w, c) to (h, w).
  Args:
    np_img: RGB Image as a NumPy array.
    output_type: Type of array to return (float or uint8)
  Returns:
    Grayscale image as NumPy array with shape (h, w).
  """
  t = Time()
  # Another common RGB ratio possibility: [0.299, 0.587, 0.114]
  grayscale = np.dot(np_img[..., :3], [0.2125, 0.7154, 0.0721])
  if output_type != "float":
    grayscale = grayscale.astype("uint8")
  np_info(grayscale, "Gray", t.elapsed())
  return grayscale


def filter_otsu_threshold(np_img, output_type="uint8"):
  """
  Compute Otsu threshold on image as a NumPy array and return binary image based on pixels above threshold.
  Args:
    np_img: Image as a NumPy array.
    output_type: Type of array to return (bool, float, or uint8).
  Returns:
    NumPy array (bool, float, or uint8) where True, 1.0, and 255 represent a pixel above Otsu threshold.
  """
  t = Time()
  otsu_thresh_value = sk_filters.threshold_otsu(np_img)
  otsu = (np_img > otsu_thresh_value)
  if output_type == "bool":
    pass
  elif output_type == "float":
    otsu = otsu.astype(float)
  else:
    otsu = otsu.astype("uint8") * 255
  np_info(otsu, "Otsu Threshold", t.elapsed())
  return otsu


########################## ADDITIONS ###############################

def apply_image_filters_light(rgb, equalize=False, slide_num=None):
    """
    Apply filters to image as NumPy array
    """
    # EQUALIZATION CAN BE VERY SLOW - IMPOSSIBLE - ON HIGH DEFINITION Level 4 to 1. HUGE RAM CONSUMPTION

    if equalize:
        rgb = filter_adaptive_equalization(rgb, nbins=256, clip_limit=0.01, output_type="bool")

    mask_not_grays = filter_grays(rgb)
    mask_remove_objects = filter_remove_small_objects(mask_not_grays, min_size=5000, output_type="bool")
    mask_remove_holes = filter_remove_small_holes(mask_remove_objects, min_size=3000, output_type="bool")

    rgb = mask_rgb(rgb, mask_remove_holes)

    return rgb


def apply_image_filters(rgb, equalize=False, slide_num=None):
    """
    Apply filters to image as NumPy array
    """
    # EQUALIZATION CAN BE VERY SLOW - IMPOSSIBLE - ON HIGH DEFINITION Level 4 to 1. HUGE RAM CONSUMPTION

    if equalize:
        rgb = filter_adaptive_equalization(rgb, nbins=256, clip_limit=0.01, output_type="bool")

    mask_not_grays = filter_grays(rgb)
    mask_remove_objects = filter_remove_small_objects(mask_not_grays, min_size=5000, output_type="bool")
    mask_remove_holes = filter_remove_small_holes(mask_remove_objects, min_size=3000, output_type="bool")

    rgb = mask_rgb(rgb, mask_remove_holes)
    np_gray = rgb2gray(rgb)
    threshold = threshold_otsu(np_gray)
    mask = np_gray > threshold
    rgb = mask_rgb(rgb, mask)

    return rgb, mask

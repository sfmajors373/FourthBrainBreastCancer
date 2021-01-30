import argparse
from datetime import datetime
import os
import numpy as np
import h5py
from skimage.filters import threshold_otsu

### DOWNLOAD THESE FROM GIT OR CLOSE THE WHOLE REPOSITORY
### https://gitlab.com/deep.TEACHING/educational-materials/tree/master/notebooks/
# medical-image-classification
from preprocessing.datamodel import SlideManager
from preprocessing.processing import split_negative_slide, split_positive_slide, rgb2gray
from preprocessing.logger import get_logger


def build_filename(slide_name, tile_size, poi, level):
    """Generate hdfs filename.

    Parameters
    ----------
    slide_name : str
        slide name.
    tile_size : int
        size of tiles used when tiling before saving in hdfs file.
    poi : float
        minimum percentage of tissue threshold needed to save a tile
    level : int
        magnification layer number of the slide.

    Returns
    -------
    string
        string built from parameters.
    """
    filename = '{}_{}x{}_poi{}_level{}.hdf5'.format(slide_name, tile_size, tile_size, poi, level)
    return os.path.join(HDFS_DIR, filename)


def store_slides_hdfs(filepath, slide_name, num_tiles_batch, tiles_batch, tile_size):
    """Create an hdfs file and fill if with tiles extracted from a slide and tiles batch info

    Parameters
    ----------
    filepath: str
        hdfs filename.
    slide_name : str
        slide name.
    num_tiles_batch : int
        number of tiles in batch saved.
    tiles_batch: np.array
        array of tiles to be stored.
    tile_size : int
        size of tiles.
    """
    # 'w-' creates file, fails if exists
    h5 = h5py.File(filepath, "w-", libver='latest')
    # creating a dataset in the file
    h5.create_dataset(slide_name,
                      (num_tiles_batch, tile_size, tile_size, 3),
                      dtype=np.uint8,
                      data=np.array(tiles_batch),
                      compression=0)
    h5.close()


def get_otsu_threshold(slide, level):
    """Return pre-calculated otsu threshold of a layer.

    Parameters
    ----------
    slide: Slide object
        WSI - Slide of interest.
    level : int
        Slide layer.

    Returns
    -------
    Otsu_threshold: float or None
        Otsu threshold of layer `level`.
    """
    # load the slide into numpy array
    arr = np.asarray(slide.get_full_slide(level=level))
    # convert it to gray scale
    arr_gray = rgb2gray(arr)
    # calculate otsu threshold
    threshold = threshold_otsu(arr_gray)
    return threshold


def generate_positive_slides(mgr, level, tile_size, poi_tumor, percent_overlap, max_tiles_per_slide,
                             early_stopping):
    """Generate hdfs files of tiles built on tumor slides.

    Parameters
    ----------
    mgr : SlideManager object
        slide manager to access all sildes stored.
    level : int
        magnification layer number of the slide.
    tile_size : int
        size of tiles used when tiling before saving in hdfs file.
    poi_tumor : float
        minimum percentage of tissue threshold needed to save a tile
    percent_overlap: float
        percentage of overlap when generating tiles. 0.5 means we next tile will overlap with 50%
        of the tile we just generated
    max_tiles_per_slide: int
        maximum number of tiles to create from a single slide
    early_stopping: int
        number of tiles to generate. If 0, ignore and goes through the whole dataset

    """
    num_slides = len(mgr.annotated_slides)
    tiles_pos = 0
    overlap = int(tile_size * percent_overlap)
    for i in range(num_slides):
        slide = mgr.annotated_slides[i]

        LOGGER.info("Working on {}".format(slide.name))
        try:
            # create a new and unconsumed tile iterator
            tile_iter = split_positive_slide(slide, level=level,
                                             tile_size=tile_size, overlap=overlap,
                                             poi_threshold=poi_tumor)

            tiles_batch = list()
            for tile, bounds in tile_iter:
                if len(tiles_batch) % 10 == 0:
                    LOGGER.info('positive slide: {}  - tiles so far: {}'.format(i,
                                                                                len(tiles_batch)))
                if len(tiles_batch) > max_tiles_per_slide:
                    break
                tiles_batch.append(tile)

            filename = build_filename(slide.name, tile_size, poi_tumor, level)
            num_tiles_batch = len(tiles_batch)

            store_slides_hdfs(filename, slide.name, num_tiles_batch, tiles_batch, tile_size)
            tiles_pos += len(tiles_batch)
            LOGGER.info('{}, {} / {}  - tiles: {}'.format(datetime.now(), i, num_slides,
                                                          len(tiles_batch)))
            LOGGER.info('positive tiles total: {}'.format(tiles_pos))

            # exit if reaching number of tiles generated aimed for
            if early_stopping > 0:
                if tiles_pos > early_stopping:
                    break

        except Exception as e:
            LOGGER.warning('slide nr {}/{} failed - {}'.format(i, num_slides, e))


def generate_negative_slides(mgr, level, tile_size, poi, percent_overlap, max_tiles_per_slide,
                             early_stopping=0):
    """Generate hdfs files of tiles built on tumor slides.

    Parameters
    ----------
    mgr : SlideManager object
        slide manager to access all sildes stored.
    level : int
        magnification layer number of the slide.
    tile_size : int
        size of tiles used when tiling before saving in hdfs file.
    poi : float
        minimum percentage of tissue threshold needed to save a tile
    percent_overlap: float
        percentage of overlap when generating tiles. 0.5 means we next tile will overlap with 50%
        of the tile we just generated
    max_tiles_per_slide: int
        maximum number of tiles to create from a single slide
    early_stopping: int
        number of tiles to generate. If 0, ignore and goes through the whole dataset

    """

    num_slides = len(mgr.negative_slides)
    tiles_neg = 0
    overlap = int(tile_size * percent_overlap)
    for i in range(num_slides):
        slide = mgr.negative_slides[i]
        LOGGER.info("Working on {}".format(slide.name))
        # try:

        threshold = get_otsu_threshold(slide, level)

        # create a new and unconsumed tile iterator
        # because we have so many  negative slides we do not use overlap
        tile_iter = split_negative_slide(slide, level=level,
                                         otsu_threshold=threshold,
                                         tile_size=tile_size, overlap=overlap,
                                         poi_threshold=poi)

        tiles_batch = list()
        for tile, bounds in tile_iter:
            if len(tiles_batch) % 10 == 0:
                LOGGER.info('negative slide: {}  - tiles so far: {}'.format(i,
                                                                            len(tiles_batch)))
            if len(tiles_batch) > max_tiles_per_slide:
                break
            tiles_batch.append(tile)

        filename = build_filename(slide.name, tile_size, poi, level)
        num_tiles_batch = len(tiles_batch)

        store_slides_hdfs(filename, slide.name, num_tiles_batch, tiles_batch, tile_size)
        tiles_neg += len(tiles_batch)
        LOGGER.info('{}, {} / {}  - tiles: {}'.format(datetime.now(), i, num_slides,
                                                      len(tiles_batch)))
        LOGGER.info('negative tiles total: {}'.format(tiles_neg))

        # exit if reaching number of tiles generated aimed for
        if early_stopping > 0:
            if tiles_neg > early_stopping:
                break

        # except Exception as e:
        #     LOGGER.warning('slide nr {}/{} failed - {}'.format(i, num_slides, e))


def main():
    global LOGGER, HDFS_DIR

    parser = argparse.ArgumentParser(description='script settings')

    parser.add_argument('--magnification_level', '-ml', dest='magnification_level', action='store',
                        default=2, type=int,
                        help='corresponds to the different magnification levels available')
    parser.add_argument('--tile_size', '-ts', dest='tile_size', action='store', default=312,
                        type=int, help='size of tiles - should be more than final tile size')
    parser.add_argument('--poi', '-p', dest='poi', action='store', default=0.2,
                        type=float, help='20% of negative tiles must contain tissue')
    parser.add_argument('--poi_tumor', '-poit', dest='poi_tumor', action='store', default=0.6,
                        type=float, help='60% of pos tiles must contain metastases')
    parser.add_argument('--percent_overlap_tumor', '-pot', dest='percent_overlap_tumor',
                        action='store', default=0.5, type=float,
                        help='to not have too few positive tile, we use half overlapping tilesize')
    parser.add_argument('--percent_overlap', '-po', dest='percent_overlap', action='store',
                        default=0.0, type=float, help='')

    parser.add_argument('--max_tiles_per_slide', '-t', dest='max_tiles_per_slide', action='store',
                        default=1000, type=int, help='max tiles generated per slide')

    parser.add_argument('--logging_file', '-f', dest='logging_file', action='store',
                        default="tiler_train", type=str, help='path the generated log file')
    parser.add_argument('--logging_level', '-l', dest='logging_level', action='store', default=1,
                        type=int, help='logging level: 1:debug - 2:warning')

    parser.add_argument('--base_directory', '-bd', dest='base_directory', action='store',
                        default='/media/nico/data/fourthbrain/project/', type=str,
                        help='raw data directory, needs to countain a training folder '
                             'then normal/tumor/lesion_annotations subfolder')
    parser.add_argument('--dataset_folder', '-df', dest='dataset_folder', action='store',
                        default='CAMELYON16', type=str, help='dataset folder name - CAMELYON16,'
                                                             ' CAMELYON17, etc')
    parser.add_argument('--output_folder', '-of', dest='output_folder', action='store',
                        default='training_CAMELYON16', type=str,
                        help='tiles folder to store hfds files')
    parser.add_argument('--num_slides_to_process', '-n', dest='num_slides_to_process',
                        action='store', default=0, type=int,
                        help='might want to limit the number of tiles to process for testing')
    parser.add_argument('--early_stopping_num', '-es', dest='early_stopping_num', action='store',
                        default=5000, type=int, help='stop script after number of tiles '
                                                      'generated reached for normal and tumor')

    args = parser.parse_args()
    tile_size = args.tile_size
    level = args.magnification_level
    poi = args.poi
    poi_tumor = args.poi_tumor
    percent_overlap_tumor = args.percent_overlap_tumor
    percent_overlap = args.percent_overlap
    max_tiles_per_slide = args.max_tiles_per_slide
    early_stopping = args.early_stopping_num
    cam_base_dir = args.base_directory
    cam16_dir = os.path.join(cam_base_dir, args.dataset_folder)
    HDFS_DIR = os.path.join(cam_base_dir, args.output_folder)
    logging_level = args.logging_level
    logging_file = os.path.join(cam_base_dir, args.logging_file)

    LOGGER = get_logger(logging_file, logging_level=logging_level)
    mgr = SlideManager(cam16_dir=cam16_dir)

    generate_positive_slides(mgr, level, tile_size, poi_tumor, percent_overlap_tumor,
                             max_tiles_per_slide, early_stopping)
    # generate_negative_slides(mgr, level, tile_size, poi, percent_overlap, max_tiles_per_slide,
    #                          early_stopping)


if __name__ == "__main__":
    main()

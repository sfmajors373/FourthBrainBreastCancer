import argparse
import os
from datetime import datetime

from preprocessing.datamodel import SlideManager
from preprocessing.processing import split_test_slide, get_otsu_threshold
from preprocessing.logger import get_logger
from preprocessing.util import build_filename, store_slides_hdfs


def generate_test_slides(mgr, level, tile_size, percent_overlap, poi, early_stopping, hdfs_dir):
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
    early_stopping: int
        number of tiles to generate. If 0, ignore and goes through the whole dataset
    hdfs_dir: str
        hdfs folder location

    """
    num_slides = len(mgr.test_slides)
    total_tiles_normal, total_tiles_tumor = 0, 0
    overlap = int(tile_size * percent_overlap)

    for i in range(num_slides):
        slide = mgr.test_slides[i]
        tumor_slide = True if slide.annotations else False
        LOGGER.info("Working on {} - Annotations = {}".format(slide.name, tumor_slide))
        threshold = get_otsu_threshold(slide, level)

        # create a new and unconsumed tile iterator
        tile_iter = split_test_slide(slide, level=level, otsu_threshold=threshold,
                                     tile_size=tile_size, overlap=overlap,
                                     poi_threshold=poi)

        tiles_normal_batch, tiles_tumor_batch = list(), list()
        for tile, bounds, label in tile_iter:
            if (len(tiles_normal_batch) + len(tiles_tumor_batch)) % 10 == 0:
                LOGGER.info('test slide: {}  - tiles so far: {}'.format(i,
                                                                        len(tiles_normal_batch) +
                                                                        len(tiles_tumor_batch)))
            if label == 1:
                tiles_tumor_batch.append(tile)
                slide_tumor_name = slide.name.replace('test', 'tumor')
            else:
                tiles_normal_batch.append(tile)
                slide_normal_name = slide.name.replace('test', 'normal')

        num_tiles_normal_batch = len(tiles_normal_batch)
        num_tiles_tumor_batch = len(tiles_tumor_batch)

        if num_tiles_normal_batch > 0:
            filename_normal = build_filename(hdfs_dir, slide_normal_name, tile_size, poi, level)
            store_slides_hdfs(filename_normal, slide_normal_name, num_tiles_normal_batch,
                              tiles_normal_batch, tile_size)
            LOGGER.info('{}, {} / {}  - tiles: {}'.format(datetime.now(), i, num_slides,
                                                          num_tiles_normal_batch))

        if num_tiles_tumor_batch > 0:
            filename_tumor = build_filename(hdfs_dir, slide_tumor_name, tile_size, poi, level)
            store_slides_hdfs(filename_tumor, slide_tumor_name, num_tiles_tumor_batch,
                              tiles_tumor_batch, tile_size)
            LOGGER.info('{}, {} / {}  - tiles: {}'.format(datetime.now(), i, num_slides,
                                                          num_tiles_tumor_batch))

        total_tiles_normal += num_tiles_normal_batch
        total_tiles_tumor += num_tiles_tumor_batch
        if early_stopping > 0:
            if min(total_tiles_tumor, total_tiles_normal) > early_stopping:
                break


def main():
    global LOGGER

    parser = argparse.ArgumentParser(description='script settings')

    parser.add_argument('--magnification_level', '-ml', dest='magnification_level', action='store',
                        default=2, type=int,
                        help='corresponds to the different magnification levels available')
    parser.add_argument('--tile_size', '-ts', dest='tile_size', action='store', default=256,
                        type=int, help='size of tiles - should be more than final tile size')
    parser.add_argument('--poi', '-p', dest='poi', action='store', default=0.2,
                        type=float, help='20% of negative tiles must contain tissue')
    parser.add_argument('--percent_overlap', '-po', dest='percent_overlap', action='store',
                        default=0.0, type=float, help='')

    parser.add_argument('--logging_file', '-f', dest='logging_file', action='store',
                        default="tiler_test", type=str, help='path the generated log file')
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
                        default='testing_CAMELYON16', type=str,
                        help='tiles folder to store hfds files')
    parser.add_argument('--num_slides_to_process', '-n', dest='num_slides_to_process',
                        action='store', default=0, type=int,
                        help='might want to limit the number of tiles to process for testing')
    parser.add_argument('--early_stopping_num', '-es', dest='early_stopping_num', action='store',
                        default=1000, type=int, help='stop script after number of tiles '
                                                      'generated reached for normal and tumor')

    args = parser.parse_args()
    tile_size = args.tile_size
    level = args.magnification_level
    poi = args.poi
    percent_overlap = args.percent_overlap
    early_stopping = args.early_stopping_num
    cam_base_dir = args.base_directory
    cam16_dir = os.path.join(cam_base_dir, args.dataset_folder)
    hdfs_dir = os.path.join(cam_base_dir, args.output_folder)
    logging_level = args.logging_level
    logging_file = os.path.join(cam_base_dir, args.logging_file)

    LOGGER = get_logger(logging_file, logging_level=logging_level)
    mgr = SlideManager(cam16_dir=cam16_dir)

    generate_test_slides(mgr, level, tile_size, poi, percent_overlap,
                         early_stopping, hdfs_dir)


if __name__ == "__main__":
    main()

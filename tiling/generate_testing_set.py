import argparse
import os

from preprocessing.datamodel import SlideManager
from preprocessing.logger import get_logger
from generate_tiles import generate_test_tiles


def main():

    parser = argparse.ArgumentParser(description='script settings')

    parser.add_argument('--magnification_level', '-ml', dest='magnification_level', action='store',
                        default=3, type=int,
                        help='corresponds to the different magnification levels available')
    parser.add_argument('--tile_size', '-ts', dest='tile_size', action='store', default=256,
                        type=int, help='size of tiles - should be more than final tile size')
    parser.add_argument('--poi', '-p', dest='poi', action='store', default=0.2,
                        type=float, help='x% of a tile must contain tissue')
    parser.add_argument('--percent_overlap', '-po', dest='percent_overlap', action='store',
                        default=0.0, type=float, help='')

    parser.add_argument('--logging_file', '-f', dest='logging_file', action='store',
                        default="tiler_test", type=str, help='path the generated log file')
    parser.add_argument('--logging_level', '-l', dest='logging_level', action='store', default=1,
                        type=int, help='logging level: 1:debug - 2:warning')

    parser.add_argument('--cam_base_dir', '-bd', dest='cam_base_dir', action='store',
                        default='/home/sarah/ForthBrainCancer-Dataset/', type=str,
                        help='raw data directory, needs to countain a training folder '
                             'then normal/tumor/lesion_annotations subfolder')
    parser.add_argument('--dataset_folder', '-df', dest='dataset_folder', action='store',
                        default='CAMELYON16', type=str, help='dataset folder name - CAMELYON16,'
                                                             ' CAMELYON17, etc')
    parser.add_argument('--output_folder', '-of', dest='output_folder', action='store',
                        default='testing_CAMELYON16', type=str,
                        help='tiles folder to store hfds files')
    parser.add_argument('--max_tiles_per_slide', '-t', dest='max_tiles_per_slide', action='store',
                        default=1000, type=int, help='max tiles generated per slide')
    parser.add_argument('--early_stopping_num', '-es', dest='early_stopping_num', action='store',
                        default=10000, type=int, help='stop script after number of tiles '
                                                     'generated reached for normal and tumor')

    args = parser.parse_args()
    tile_size = args.tile_size
    level = args.magnification_level
    poi = args.poi
    percent_overlap = args.percent_overlap
    early_stopping = args.early_stopping_num
    max_tiles_per_slide = args.max_tiles_per_slide
    cam_base_dir = args.cam_base_dir
    cam16_dir = os.path.join(cam_base_dir, args.dataset_folder)
    hdfs_dir = os.path.join(cam_base_dir, args.output_folder)
    logging_level = args.logging_level
    logging_file = os.path.join(cam_base_dir, args.logging_file)

    logger = get_logger(logging_file, logging_level=logging_level)
    mgr = SlideManager(cam16_dir=cam16_dir)
    print('MGR: ', len(mgr.test_slides))
    print('About to start')

    generate_test_tiles(mgr, level, tile_size, poi, percent_overlap, max_tiles_per_slide,
                        logger, hdfs_dir, early_stopping)


if __name__ == "__main__":
    main()

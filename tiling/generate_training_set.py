import argparse
import os

from preprocessing.datamodel import SlideManager
from preprocessing.logger import get_logger
from generate_tiles import generate_negative_tiles, generate_positive_tiles


def main():

    parser = argparse.ArgumentParser(description='script settings')

    parser.add_argument('--magnification_level', '-ml', dest='magnification_level', action='store',
                        default=3, type=int,
                        help='corresponds to the different magnification levels available')
    parser.add_argument('--tile_size', '-ts', dest='tile_size', action='store', default=312,
                        type=int, help='size of tiles - should be more than final tile size')
    parser.add_argument('--poi', '-p', dest='poi', action='store', default=0.2,
                        type=float, help='x% of negative tiles must contain tissue')
    parser.add_argument('--poi_tumor', '-poit', dest='poi_tumor', action='store', default=0.4,
                        type=float, help='x% of pos tiles must contain metastases')
    parser.add_argument('--percent_overlap_tumor', '-pot', dest='percent_overlap_tumor',
                        action='store', default=0.5, type=float,
                        help='to not have too few positive tile, we use half overlapping tilesize')
    parser.add_argument('--percent_overlap', '-po', dest='percent_overlap', action='store',
                        default=0.0, type=float, help='')
    parser.add_argument('--use_upstream_filters', '-uuf', dest='use_upstream_filters',
                        action='store', default=False, type=bool, help='')
    parser.add_argument('--max_tiles_per_slide', '-t', dest='max_tiles_per_slide', action='store',
                        default=1000, type=int, help='max tiles generated per slide')
    parser.add_argument('--logging_file', '-f', dest='logging_file', action='store',
                        default="tiler_train", type=str, help='path the generated log file')
    parser.add_argument('--logging_level', '-l', dest='logging_level', action='store', default=1,
                        type=int, help='logging level: 1:debug - 2:warning')
    parser.add_argument('--base_directory', '-bd', dest='base_directory', action='store',
                        default='/media/nico/data/fourthbrain/project/', type=str,
                        help='raw data directory, needs to contain a training folder '
                             'then normal/tumor/lesion_annotations subfolder')
    parser.add_argument('--dataset_folder', '-df', dest='dataset_folder', action='store',
                        default='CAMELYON16', type=str, help='dataset folder name - CAMELYON16,'
                                                             ' CAMELYON17, etc')
    parser.add_argument('--output_folder', '-of', dest='output_folder', action='store',
                        default='training_CAMELYON16', type=str,
                        help='tiles folder to store hfds files')
    parser.add_argument('--early_stopping_num', '-es', dest='early_stopping_num', action='store',
                        default=10000, type=int, help='stop script after number of tiles '
                                                     'generated reached for normal and tumor')

    args = parser.parse_args()
    tile_size = args.tile_size
    level = args.magnification_level
    poi = args.poi
    poi_tumor = args.poi_tumor
    percent_overlap_tumor = args.percent_overlap_tumor
    percent_overlap = args.percent_overlap
    use_upstream_filters = args.use_upstream_filters
    max_tiles_per_slide = args.max_tiles_per_slide
    early_stopping = args.early_stopping_num
    cam_base_dir = args.base_directory
    cam16_dir = os.path.join(cam_base_dir, args.dataset_folder)
    hdfs_dir = os.path.join(cam_base_dir, args.output_folder)
    logging_level = args.logging_level
    logging_file = os.path.join(cam_base_dir, args.logging_file)

    logger = get_logger(logging_file, logging_level=logging_level)
    mgr = SlideManager(cam16_dir=cam16_dir)

    generate_positive_tiles(mgr, level, tile_size, poi_tumor, percent_overlap_tumor,
                            max_tiles_per_slide, logger, hdfs_dir, early_stopping)
    generate_negative_tiles(mgr, level, tile_size, poi, percent_overlap, max_tiles_per_slide,
                            use_upstream_filters, logger, hdfs_dir, early_stopping)


if __name__ == "__main__":
    main()

import argparse
import os
import numpy as np

from read_tiles import TissueDataset
from preprocessing.logger import get_logger
from preprocessing.util import save_color_normalization_values


def generate_color_normalization(num_negative_tiles, num_positive_tiles):

    train_data = TissueDataset(HDFS_DIR,  percentage=1.0, first_part=False)
    x, _ = train_data.get_batch(num_neg=num_negative_tiles, num_pos=num_positive_tiles)
    list_r, list_g, list_b = list(), list(), list()
    for np_img in x:
        list_r.append(np_img[:, :, 0])
        list_g.append(np_img[:, :, 1])
        list_b.append(np_img[:, :, 2])

    r = np.concatenate(list_r).flatten()
    g = np.concatenate(list_g).flatten()
    b = np.concatenate(list_b).flatten()

    mean = [r.mean(), g.mean(), b.mean()]
    std = [r.std(), g.std(), b.std()]

    LOGGER.info("mean: {}, std={}".format(mean, std))

    save_color_normalization_values(mean, std, filename=JSON_OUTPUT)
    LOGGER.info("json saved at: {}".format(JSON_OUTPUT))


def main():
    global LOGGER, HDFS_DIR, JSON_OUTPUT

    parser = argparse.ArgumentParser(description='script settings')

    parser.add_argument('--base_directory', '-bd', dest='base_directory', action='store',
                        default='/media/nico/data/fourthbrain/project/', type=str,
                        help='raw data directory, needs to countain a training folder '
                             'then normal/tumor/lesion_annotations subfolder')
    parser.add_argument('--hdfs_folder', '-hf', dest='hdfs_folder', action='store',
                        default='output_CAMELYON16', type=str,
                        help='tiles folder to store hfds files')
    parser.add_argument('--output_file', '-of', dest='output_file', action='store',
                        default='CAMELYON16_color_normalization.json', type=str, help='')

    parser.add_argument('--num_negative_tiles', '-n', dest='num_negative_tiles', action='store',
                        default=2500, type=int, help='number of tiles from negative slides')
    parser.add_argument('--num_positive_tiles', '-n', dest='num_positive_tiles', action='store',
                        default=2500, type=int, help='number of tiles from positive slides')

    parser.add_argument('--logging_file', '-f', dest='logging_file', action='store',
                        default="color_norm_json", type=str, help='path the generated log file')
    parser.add_argument('--logging_level', '-l', dest='logging_levbase_directoryel', action='store',
                        default=1, type=int, help='logging level: 1:debug - 2:warning')

    args = parser.parse_args()
    cam_base_dir = args.base_directory
    HDFS_DIR = os.path.join(cam_base_dir, args.hdfs_folder)
    JSON_OUTPUT = os.path.join(cam_base_dir, args.output_file)
    logging_level = args.logging_level
    logging_file = os.path.join(cam_base_dir, args.logging_file)

    LOGGER = get_logger(logging_file, logging_level=logging_level)

    generate_color_normalization(args.num_negative_tiles, args.num_positive_tiles)


if __name__ == "__main__":
    main()

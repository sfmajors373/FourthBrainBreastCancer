import argparse
from datetime import datetime
import os
import numpy as np
import h5py
from skimage.filters import threshold_otsu

### DOWNLOAD THESE FROM GIT OR CLOSE THE WHOLE REPOSITORY
### https://gitlab.com/deep.TEACHING/educational-materials/tree/master/notebooks/medical-image-classification
from preprocessing.datamodel import SlideManager
from preprocessing.processing import split_negative_slide, split_positive_slide, rgb2gray
from preprocessing.logger import get_logger


def build_filename(slide_name, tile_size, poi, level):
    filename = '{}_{}x{}_poi{}_level{}.hdf5'.format(slide_name, tile_size, tile_size, poi, level)
    return os.path.join(HDFS_DIR, filename)


def store_slides_hdfs(filepath, slide_name, num_tiles_batch, tiles_batch, tile_size):
    # 'w-' creates file, fails if exists
    h5 = h5py.File(filepath, "w-", libver='latest')
    # creating a dataset in the file
    h5.create_dataset(slide_name,
                      (num_tiles_batch, tile_size, tile_size, 3),
                      dtype=np.uint8,
                      data=np.array(tiles_batch),
                      compression=0)
    h5.close()


def generate_positive_slides(mgr, level, tile_size, poi_tumor, percent_overlap, max_tiles_per_slide, early_stopping):
    num_slides = len(mgr.annotated_slides)
    tiles_pos = 0
    overlap = int(tile_size * percent_overlap)
    for i in range(num_slides):
        slide_name = mgr.annotated_slides[i].name
        LOGGER.info("Working on {}".format(slide_name))
        try:
            # create a new and unconsumed tile iterator
            tile_iter = split_positive_slide(mgr.annotated_slides[i], level=level,
                                             tile_size=tile_size, overlap=overlap,
                                             poi_threshold=poi_tumor)

            tiles_batch = list()
            for tile, bounds in tile_iter:
                if len(tiles_batch) % 10 == 0:
                    LOGGER.info('positive slide: {}  - tiles so far: {}'.format(i, len(tiles_batch)))
                if len(tiles_batch) > max_tiles_per_slide : break
                tiles_batch.append(tile)

            filename = build_filename(slide_name, tile_size, poi_tumor, level)
            num_tiles_batch = len(tiles_batch)

            store_slides_hdfs(filename, slide_name, num_tiles_batch, tiles_batch, tile_size)
            tiles_pos += len(tiles_batch)
            LOGGER.info('{}, {} / {}  - tiles: {}'.format(datetime.now(), i, num_slides, len(tiles_batch)))
            LOGGER.info('pos tiles total: {}'.format(tiles_pos))

            # exit if reaching number of tiles generated aimed for
            if early_stopping > 0:
                if tiles_pos > early_stopping:
                    break

        except Exception as e:
            LOGGER.warning('slide nr {}/{} failed - {}'.format(i, num_slides, e))


def generate_negative_slides(mgr, level, tile_size, poi, percent_overlap, max_tiles_per_slide, early_stopping):
    num_slides = len(mgr.negative_slides)
    tiles_neg = 0
    overlap = int(tile_size * percent_overlap)
    for i in range(num_slides):
        slide_name = mgr.negative_slides[i].name
        LOGGER.info("Working on {}".format(slide_name))
        try:

            # load the slide into numpy array
            arr = np.asarray(mgr.negative_slides[i].get_full_slide(level=4))
            # convert it to gray scale
            arr_gray = rgb2gray(arr)
            # calculate otsu threshold
            threshold = threshold_otsu(arr_gray)

            # create a new and unconsumed tile iterator
            # because we have so many  negative slides we do not use overlap
            tile_iter = split_negative_slide(mgr.negative_slides[i], level=level,
                                             otsu_threshold=threshold,
                                             tile_size=tile_size, overlap=overlap,
                                             poi_threshold=poi)

            tiles_batch = list()
            for tile, bounds in tile_iter:
                if len(tiles_batch) % 10 == 0:
                    LOGGER.info('negative slide: {}  - tiles so far: {}'.format(i, len(tiles_batch)))
                if len(tiles_batch) > max_tiles_per_slide:
                    break
                tiles_batch.append(tile)

            filename = build_filename(slide_name, tile_size, poi, level)
            num_tiles_batch = len(tiles_batch)

            store_slides_hdfs(filename, slide_name, num_tiles_batch, tiles_batch, tile_size)
            tiles_neg += len(tiles_batch)
            LOGGER.info('{}, {} / {}  - tiles: {}'.format(datetime.now(), i, num_slides, len(tiles_batch)))
            LOGGER.info('neg tiles total: ', tiles_neg)

            # exit if reaching number of tiles generated aimed for
            if early_stopping > 0:
                if tiles_neg > early_stopping:
                    break

        except Exception as e:
            LOGGER.warning('slide nr {}/{} failed - {}'.format(i, num_slides, e))


def main():
    global LOGGER, CAM_BASE_DIR, HDFS_DIR

    parser = argparse.ArgumentParser(description='script settings')

    parser.add_argument('--magnification_level', '-ml', dest='magnification_level', action='store', default=1,
                        type=int, help='1 to 8')
    parser.add_argument('--tile_size', '-ts', dest='tile_size', action='store', default=256,
                        type=int, help='1 to 8')
    parser.add_argument('--poi', '-p', dest='poi', action='store', default=0.2,
                        type=float, help='20% of negative tiles must contain tissue (in contrast to slide background)')
    parser.add_argument('--poi_tumor', '-poit', dest='poi_tumor', action='store', default=0.6,
                        type=float, help='60% of pos tiles must contain metastases')
    parser.add_argument('--percent_overlap_tumor', '-pot', dest='percent_overlap_tumor', action='store', default=0.5,
                        type=float, help='to not have too few positive tile, we use half overlapping tilesize')
    parser.add_argument('--percent_overlap', '-po', dest='percent_overlap', action='store', default=0.0,
                        type=float, help='')

    parser.add_argument('--max_tiles_per_slide', '-t', dest='max_tiles_per_slide', action='store', default=1000,
                        type=int, help='max tiles generated per slide')

    parser.add_argument('--logging_file', '-f', dest='logging_file', action='store', default="tiler",
                        type=str, help='path the generated log file')
    parser.add_argument('--logging_level', '-l', dest='logging_level', action='store', default=1,
                        type=int, help='logging level: 1:debug - 2:warning')

    parser.add_argument('--base_directory', '-bd', dest='base_directory', action='store', default='/media/nico/data/fourthbrain/project/',
                        type=str, help='raw data directory, needs to countain a training folder then normal/tumor/lesion_annotations subfolder')
    parser.add_argument('--dataset_folder', '-df', dest='dataset_folder', action='store', default='CAMELYON16',
                        type=str, help='dataset folder name - CAMELYON16, CAMELYON17, etc')
    parser.add_argument('--output_folder', '-of', dest='output_folder', action='store', default='output_CAMELYON16',
                        type=str, help='tiles folder to store hfds files')
    parser.add_argument('--num_slides_to_process', '-n', dest='num_slides_to_process', action='store', default=0,
                        type=int, help='might want to limit the number of tiles to process for testing')
    parser.add_argument('--early_stopping_num', '-es', dest='early_stopping_num', action='store', default=1000,
                        type=int, help='stop script after number of tiles generated reached for normal and tumor')

    args = parser.parse_args()
    tile_size = args.tile_size
    level = args.magnification_level
    poi = args.poi
    poi_tumor = args.poi_tumor
    percent_overlap_tumor = args.percent_overlap_tumor
    percent_overlap = args.percent_overlap
    max_tiles_per_slide = args.max_tiles_per_slide
    early_stopping = args.early_stopping_num
    CAM_BASE_DIR = args.base_directory
    CAM16_DIR = os.path.join(CAM_BASE_DIR, args.dataset_folder)
    HDFS_DIR = os.path.join(CAM_BASE_DIR, args.output_folder)
    logging_level = args.logging_level
    logging_file = os.path.join(CAM_BASE_DIR, args.logging_file)

    LOGGER = get_logger(logging_file, logging_level=logging_level)
    mgr = SlideManager(cam16_dir=CAM16_DIR)

    generate_positive_slides(mgr, level, tile_size, poi_tumor, percent_overlap_tumor, max_tiles_per_slide, early_stopping)
    generate_negative_slides(mgr, level, tile_size, poi, percent_overlap, max_tiles_per_slide, early_stopping)


if __name__ == "__main__":
    main()

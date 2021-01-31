import math
import numpy as np
import os
from tiling.preprocessing.util import combine_datasets, load_color_normalization_values, hdfs_filepaths
import h5py


def data_filenames(folder):
    """Read files path in folder and return list of positive and negative ones

    Parameters
    ----------
    folder : str
        folder path.

    Returns
    -------
    pos : list
        list of all hdfs files full paths built from positive (tumor) slides.
    neg : list
        list of all hdfs files full paths built from negative (normal) slides.
    tile_size : int
        tile size of tiles stored in hdfs folder
    """

    masks, tiles = list(), list()
    filenames = hdfs_filepaths(folder)
    for filename in filenames:
        data_file = h5py.File(filename, 'r', libver='latest', swmr=True)
        # we only have one key as we separate slides
        key = list(data_file.keys())[0]
        data_shape = data_file[key].shape
        # if no tiles were stored - shape (0, x, x, 3) pass
        if data_shape[0] == 0:
            continue
        tile_size = data_shape[1]
        if 'mask' in filename:
            masks.append(filename)
        else:
            tiles.append(filename)
    return sorted(masks), sorted(tiles), tile_size


class TissueDataset:
    """Data set for preprocessed WSIs of the CAMELYON16 and CAMELYON17 data set."""

    def __init__(self, folder, percentage=.5, first_part=True, crop_size=256):
        self.h5_folder = folder
        self.perc = percentage
        self.first_part = first_part
        self.masks, self.tiles, self.tile_size = data_filenames(folder)
        self.crop_size = crop_size
        self.tiles_dataset = combine_datasets(self.tiles)
        self.masks_dataset = combine_datasets(self.masks)

    def __get_tiles_from_path(self, dataset_tiles, dataset_masks, number_tiles):
        tiles = np.ndarray((number_tiles, self.crop_size, self.crop_size, 3))
        masks = np.ndarray((number_tiles, self.crop_size, self.crop_size, 1))

        for i in range(number_tiles):
            len_ds = len(dataset_tiles)
            max_tiles = math.ceil(len_ds * self.perc)
            if self.first_part:
                rnd_idx = np.random.randint(0, max_tiles)
            else:
                rnd_idx = np.random.randint(len_ds - max_tiles, len_ds)
            ### crop random crop_size x crop_size
            if self.tile_size > self.crop_size:
                rand_height = np.random.randint(0, self.tile_size - self.crop_size)
                rand_width = np.random.randint(0, self.tile_size - self.crop_size)
            else:
                rand_height = 0
                rand_width = 0
            tiles[i] = dataset_tiles[rnd_idx, rand_height:rand_height + self.crop_size,
                       rand_width:rand_width + self.crop_size]
            masks[i] = dataset_masks[rnd_idx, rand_height:rand_height + self.crop_size,
                       rand_width:rand_width + self.crop_size]

        tiles = tiles / 255.

        return tiles, masks

    def __get_random_positive_tiles(self, number_tiles):
        return self.__get_tiles_from_path(self.tiles_dataset, self.masks_dataset, number_tiles)

    # def __get_random_negative_tiles(self, number_tiles):
    #     return self.__get_tiles_from_path(self.neg_dataset, number_tiles), np.zeros((number_tiles))

    def generator(self, num_neg=10, data_augm=False,
                  color_normalization_file="CAMELYON16_color_normalization.json",
                  green_layer_only=False):

        mean, std = load_color_normalization_values(color_normalization_file)

        while True:
            tile, mask = self.get_batch(num_neg, data_augm)
            for i in [0, 1, 2]:
                tile[:, :, :, i] = (tile[:, :, :, i] - mean[i]) / std[i]

            if green_layer_only:
                tile[:, :, :, 0] = 1.
                tile[:, :, :, 0] = 1.

            yield tile, mask

    def get_batch(self, num_neg=10, data_augm=False):

        # x_p, y_p = self.__get_random_positive_tiles(num_pos)
        x_n, y_n = self.__get_random_positive_tiles(num_neg)

        x = np.asarray(x_n) #np.concatenate((x_p, x_n), axis=0)
        y = np.asarray(y_n) #np.concatenate((y_p, y_n), axis=0)

        if data_augm:
            ### some data augmentation mirroring / rotation
            if np.random.randint(0, 2):
                x = np.flip(x, axis=1)
                y = np.flip(y, axis=1)
            if np.random.randint(0, 2):
                x = np.flip(x, axis=2)
                y = np.flip(y, axis=2)
            k = np.random.randint(0, 4)
            x = np.rot90(m=x, k=k, axes=(1, 2))
            y = np.rot90(m=y, k=k, axes=(1, 2))

        ### randomly arrange in order
        p = np.random.permutation(len(y))

        return x[p], y[p]


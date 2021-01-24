import math
import numpy as np

from tiling.preprocessing.util import pos_neg_filenames, combine_datasets, load_color_normalization_values


class TissueDataset:
    """Data set for preprocessed WSIs of the CAMELYON16 and CAMELYON17 data set."""

    def __init__(self, folder, percentage=.5, first_part=True, crop_size=256):
        # this is different from the original deep.teaching code
        # this script assumes we have concatenated all the hdfs files generated in tiling
        # into 1 file for now, we are going to keep each hdfs file generated separate
        self.h5_folder = folder
        self.perc = percentage
        self.first_part = first_part
        self.neg, self.pos, self.tile_size = pos_neg_filenames(folder)
        self.crop_size = crop_size
        self.neg_dataset = combine_datasets(self.neg)
        self.pos_dataset = combine_datasets(self.pos)

    def __get_tiles_from_path(self, dataset, number_tiles):
        tiles = np.ndarray((number_tiles, self.crop_size, self.crop_size, 3))
        for i in range(number_tiles):
            len_ds = len(dataset)
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
            tiles[i] = dataset[rnd_idx, rand_height :rand_height + self.crop_size,
                       rand_width:rand_width + self.crop_size]
        tiles = tiles / 255.
        return tiles

    def __get_random_positive_tiles(self, number_tiles):
        return self.__get_tiles_from_path(self.pos_dataset, number_tiles), np.ones((number_tiles))

    def __get_random_negative_tiles(self, number_tiles):
        return self.__get_tiles_from_path(self.neg_dataset, number_tiles), np.zeros((number_tiles))

    def generator(self, num_neg=10, num_pos=10, data_augm=False,
                  color_normalization_file="CAMELYON16_color_normalization.json",
                  green_layer_only=False):

        mean, std = load_color_normalization_values(color_normalization_file)

        while True:
            tile, label = self.get_batch(num_neg, num_pos, data_augm)
            for i in [0, 1, 2]:
                tile[:, :, :, i] = (tile[:, :, :, i] - mean[i]) / std[i]

            if green_layer_only:
                tile = np.dot(tile[..., :3], [0.0, 1.0, 0.0])

            yield tile, label

    def get_batch(self, num_neg=10, num_pos=10, data_augm=False):
        x_p, y_p = self.__get_random_positive_tiles(num_pos)
        x_n, y_n = self.__get_random_negative_tiles(num_neg)
        x = np.concatenate((x_p, x_n), axis=0)
        y = np.concatenate((y_p, y_n), axis=0)
        if data_augm:
            ### some data augmentation mirroring / rotation
            if np.random.randint(0, 2):
                x = np.flip(x, axis=1)
            if np.random.randint(0, 2):
                x = np.flip(x, axis=2)
            x = np.rot90(m=x, k=np.random.randint(0, 4), axes=(1, 2))
        ### randomly arrange in order
        p = np.random.permutation(len(y))
        return x[p], y[p]


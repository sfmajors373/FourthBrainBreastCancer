import numpy as np
import h5py
import math


class TissueDataset:
    """Data set for preprocessed WSIs of the CAMELYON16 and CAMELYON17 data set."""

    def __init__(self, path, percentage=.5, first_part=True, crop_size=256):
        self.h5_file = path
        self.h5 = h5py.File(path, 'r', libver='latest', swmr=True)
        self.perc = percentage
        self.first_part = first_part
        self.dataset_names = list(self.h5.keys())
        self.neg = [i for i in self.dataset_names if 'ormal' in i]
        self.pos = [i for i in self.dataset_names if 'umor' in i]
        self.dims = self.h5[self.neg[0]][0].shape
        self.crop_size = crop_size

    def __get_tiles_from_path(self, dataset_names, max_wsis, number_tiles):
        tiles = np.ndarray((number_tiles, self.crop_size, self.crop_size, 3))
        for i in range(number_tiles):
            file_idx = np.random.randint(0, max_wsis)
            dset = self.h5[dataset_names[file_idx]]
            len_ds = len(dset)
            max_tiles = math.ceil(len_ds * self.perc)
            if self.first_part:
                rnd_idx = np.random.randint(0, max_tiles)
            else:
                rnd_idx = np.random.randint(len_ds - max_tiles, len_ds)
            ### crop random crop_size x crop_size
            if self.dims[1] > self.crop_size:
                rand_height = np.random.randint(0, self.dims[0] - self.crop_size)
                rand_width = np.random.randint(0, self.dims[1] - self.crop_size)
            else:
                rand_height = 0
                rand_width = 0
            tiles[i] = dset[rnd_idx, rand_height:rand_height + self.crop_size, rand_width:rand_width + self.crop_size]
        tiles = tiles / 255.
        return tiles

    def __get_random_positive_tiles(self, number_tiles):
        return self.__get_tiles_from_path(self.pos, len(self.pos), number_tiles), np.ones((number_tiles))

    def __get_random_negative_tiles(self, number_tiles):
        return self.__get_tiles_from_path(self.neg, len(self.neg), number_tiles), np.zeros((number_tiles))

    def generator(self, num_neg=10, num_pos=10, data_augm=False, mean=[0., 0., 0.], std=[1., 1., 1.]):
        while True:
            x, y = self.get_batch(num_neg, num_pos, data_augm)
            for i in [0, 1, 2]:
                x[:, :, :, i] = (x[:, :, :, i] - mean[i]) / std[i]
            yield x, y

    def get_batch(self, num_neg=10, num_pos=10, data_augm=False):
        x_p, y_p = self.__get_random_positive_tiles(num_pos)
        x_n, y_n = self.__get_random_negative_tiles(num_neg)
        x = np.concatenate((x_p, x_n), axis=0)
        y = np.concatenate((y_p, y_n), axis=0)
        if data_augm:
            ### some data augmentation mirroring / rotation
            if np.random.randint(0, 2): x = np.flip(x, axis=1)
            if np.random.randint(0, 2): x = np.flip(x, axis=2)
            x = np.rot90(m=x, k=np.random.randint(0, 4), axes=(1, 2))
        ### randomly arrange in order
        p = np.random.permutation(len(y))
        return x[p], y[p]


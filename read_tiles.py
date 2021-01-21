import numpy as np
import h5py
import math
from os import listdir
from os.path import isfile, join


def hdfs_filepaths(folder):
    return [join(folder, f) for f in listdir(folder) if isfile(join(folder, f))]

def pos_neg_filenames(folder):
    pos, neg = list(), list()
    files = hdfs_filepaths(folder)
    for f in files:
        data_file = h5py.File(f,'r',libver='latest',swmr=True)
        # we only have one key as we separate slides
        key = list(data_file.keys())[0]
        data_shape = data_file[key].shape
        # if no tiles were stored - shape (0, x, x, 3) pass
        if data_shape[0] == 0:
            continue
        tile_size = data_shape[1]
        if 'tumor' in f:
            neg.append(f)
        elif 'normal' in f:
            pos.append(f)
    return pos, neg, tile_size


def combine_datasets(filenames):
    i = 0
    for f in filenames:
        data_file = h5py.File(f,'r',libver='latest',swmr=True)
        key = list(data_file.keys())[0]
        if i == 0:
            dset = data_file[key]
        else:
            dset = np.concatenate((dset, data_file[key]), axis=0)
        i += 1
    return dset


class TissueDataset:
    """Data set for preprocessed WSIs of the CAMELYON16 and CAMELYON17 data set."""

    def __init__(self, folder, percentage=.5, first_part=True, crop_size=256):
        # this is different from the original deep.teaching code
        # this script assumes we have concatenated all the hdfs files generated in tiling into 1 file
        # for now, we are going to keep each hdfs file generated separate
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
            ### crop random 256x256
            if self.tile_size > self.crop_size:
                rand_height = np.random.randint(0, self.tile_size - self.crop_size)
                rand_width = np.random.randint(0, self.tile_size - self.crop_size)
            else:
                rand_height = 0
                rand_width = 0
            tiles[i] = dataset[rnd_idx, rand_height:rand_height+256, rand_width:rand_width+256]
        tiles = tiles / 255.
        return tiles

    def __get_random_positive_tiles(self, number_tiles):
        return self.__get_tiles_from_path(self.pos_dataset, number_tiles), np.ones((number_tiles))

    def __get_random_negative_tiles(self, number_tiles):
        return self.__get_tiles_from_path(self.neg_dataset, number_tiles), np.zeros((number_tiles))

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


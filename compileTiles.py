import os
import h5py

def compileFiles(SINGLE_FILE, GENERATED_DATA):
    h5_single = h5py.File(GENERATED_DATA + SINGLE_FILE, 'w')
    for f in os.listdir(GENERATED_DATA):
        if f.startswith('normal_') or f.startswith('tumor'):
            filename = GENERATED_DATA + f
            with h5py.File(filename, 'r') as h5:
                for key in h5.keys():
                    print('processing: "{}", shape: {}'.format(key, h5[key].shape))
                    if h5[key].shape[0] > 0:
                        dset = h5_single.create_dataset(key,
                                h5[key].shape,
                                dtype=np.uint8,
                                data=h5[key][:],
                                compression=0)
    h5_single.close()

def main():

    parser = argparse.ArgumentParser(description='script settings')

    parser.add_argument('--path_to_data', '-pd', dest='path_to_data', action='store',
            default='/home/sarah/ForthBrainCancer-Dataset/training_CAMELYON/',
            help='path to generated tiles')
    parser.add_argument('--file_name', '-fn', dest='file_name', action='store',
            default='all_wsis_312x312_poi0.4_level3.hdf5',
            help='name of file')

    args = parser.parse_args()
    GENERATED_DATA = args.path_to_data
    SINGLE_FILE = args.file_name

    compileFiles(SINGLE_FILE, GENERATED_DATA)

if __name__ == '__main__':
    main()


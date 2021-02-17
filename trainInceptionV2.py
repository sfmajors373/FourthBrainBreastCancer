import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import random
import h5py
import math
from datetime import datetime
import argparse

from preprocessing.util import find_files
from preprocessing.datamodel import SlideManager
from preprocessing.util import TileMap
from preprocessing.originalTissueDS import TissueDataset

def main():

    parser = argparse.ArgumentParser(description='script settings')

    parser.add_argument('--single_file_path', '-sf', dest='single_file_path', action='store',
            default='/home/sarah/ForthBrainCancer-Dataset/all_wsis_312x312_poi0.4_level3.hdf5',
            type=str, help='file to compiled file of training data in hdf5 format')
    parser.add_argument('--model_final_path', '-mf', dest='model_final_path', action='store',
            default='',
            type=str, help='path to file to save model in hdf5 format')
    parser.add_argument('--model_checkpoint', '-cp', dest='model_checkpoint', action='store',
            default='',
            type=str, help='path to file to save periodic checkpoints of model weights')
    parser.add_argument('--pickle', '-pf', dest='pickle', action='store',
            default='',
            type=str, help='path to save final model in pickle format')


    args=parser.parse_args()
    SINGLE_FILE = args.single_file_path
    MODEL_FINAL = args.model_final_path
    MODEL_CHECKPOINT = args.model_checkpoint
    PICKLE_FILE = args.pickle

    model.save(MODEL_FINAL)train_data = TissueDataset(path=SINGLE_FILE, percentage=0.5, first_part=True)
    val_data = TissueDataset(path=SINGLE_FILE, percentage=0.5, first_part=False)

    x, y = train_data.get_batch(num_neg=3, num_pos=3)

    base_model = keras.applications.InceptionResNetV2(
                                    include_top=False,
                                    weights='imagenet',
                                    input_shape=(256,256,3),
                                    )

    x = base_model.output
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(1024, activation='relu')(x)
    predictions = keras.layers.Dense(1, activation='sigmoid')(x)

    model = keras.Model(inputs=base_model.input, outputs=predictions)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                loss='binary_crossentropy',
                metrics=['accuracy'])

    train_accs = []
    train_losses = []
    val_accs = []
    val_losses = []

    batch_size_neg=20
    batch_size_pos=20
    batches_per_train_epoch = 100
    batches_per_val_epoch = 50
    epochs = 50

    cp_callback = tf.keras.callbacks.ModelCheckpoint(MODEL_CHECKPOINT,
                                                    save_weights_only=True,
                                                    save_best_only=True,
                                                    verbose=1)

    now1 = datetime.now()

    ### Uncomment to easily disable color normalization
    mean_pixel = [0.,0.,0.]
    std_pixel = [1.,1.,1.]

    for i in range(epochs):
        hist = model.fit_generator(
                generator=train_data.generator(batch_size_neg, batch_size_pos, True, mean_pixel, std_pixel),
                steps_per_epoch=batches_per_train_epoch,
                validation_data=val_data.generator(batch_size_neg, batch_size_pos, False, mean_pixel, std_pixel),
                validation_steps=batches_per_val_epoch,
                callbacks=[cp_callback], workers=1, use_multiprocessing=False, max_queue_size=10)

        train_accs.append(hist.history['accuracy'])
        train_losses.append(hist.history['loss'])
        val_accs.append(hist.history['val_accuracy'])
        val_losses.append(hist.history['val_loss'])

    now2 = datetime.now()
    print(now2 - now1)

    model.save(MODEL_FINAL)
    pickle.dump(model, open(PICKLE_FILE, 'wb'))

if __name__ == '__main__':
    main()

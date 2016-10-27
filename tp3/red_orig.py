from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility
import os
from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from scipy import misc
from time import time
import matplotlib.pyplot as plt

from keras.preprocessing.image import ImageDataGenerator
import shutil

batch_size = 128
nb_classes = 91
nb_epoch = 12

LOAD_MODEL = False

# input image dimensions
img_rows, img_cols = 32, 32
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (3, 3)

def prepare_show(face):
    m = face.min()
    M = face.max()
    return (face-m)/(M-m)

imgDataGen = ImageDataGenerator(featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    rotation_range=0.,
    width_shift_range=0.,
    height_shift_range=0.,
    shear_range=0.,
    zoom_range=0.,
    channel_shift_range=0.,
    fill_mode='nearest',
    cval=0.,
    
    horizontal_flip=False,
    vertical_flip=False,
    rescale=1/255.,
    dim_ordering=K.image_dim_ordering())

train_generator = imgDataGen.flow_from_directory("dataset/train", target_size=(img_rows, img_cols), color_mode='grayscale', batch_size=batch_size)
test_generator = imgDataGen.flow_from_directory("dataset/test", target_size=(img_rows, img_cols), color_mode='grayscale',  batch_size=batch_size)
valid_generator = imgDataGen.flow_from_directory("dataset/valid", target_size=(img_rows, img_cols), color_mode='grayscale',  batch_size=batch_size)

if K.image_dim_ordering() == 'th':
    input_shape = (1, img_rows, img_cols)
else:
    input_shape = (img_rows, img_cols, 1)

nombre_red = os.path.basename(__file__) + '-' + str(int(time()))

if LOAD_MODEL:
    print("Cargando modelo...")
    model = load_model("red_orig.py-model-1477598524.h5")
else:
    print("Armando modelo...")
    model = Sequential()

    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                            border_mode='valid',
                            input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    print("Compilando...")
    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])
       
    print("Entrenando red: "+nombre_red)
    history = model.fit_generator(train_generator, samples_per_epoch=269018, nb_epoch=nb_epoch,
              verbose=1, validation_data=valid_generator, nb_val_samples=25000)

    file_name = nombre_red+'-model.h5'
    print("Guardando pesos en "+file_name+"...")
    model.save(file_name)

    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train', 'val'], loc='lower right')
    plt.savefig(nombre_red+'-acc.png', bbox_inches='tight', dpi = 150)
    plt.clf()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'val'], loc='upper right')
    plt.savefig(nombre_red+'-loss.png', bbox_inches='tight', dpi = 150)
    plt.clf()

print("Evaluando modelo...")
score = model.evaluate_generator(train_generator, val_samples=50000)
print('Train score:', score[0])
print('Train accuracy:', score[1])
score = model.evaluate_generator(valid_generator, val_samples=50000)
print('Valid score:', score[0])
print('Valid accuracy:', score[1])
score = model.evaluate_generator(test_generator, val_samples=50000)
print('Test score:', score[0])
print('Test accuracy:', score[1])

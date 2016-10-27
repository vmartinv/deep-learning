'''Trains a simple convnet on the MNIST dataset.
Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility
import os
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from scipy import misc
from time import time

from keras.preprocessing.image import ImageDataGenerator
import shutil

batch_size = 128
nb_classes = 91
nb_epoch = 12

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
    
# the data, shuffled and split between train and test sets
def get_images(base, tam=None):
    print("Cargando "+base+"...")
    extension = '.png'
    files = []
    for clase in os.listdir(base):
        inputdir = os.path.join(base, clase)
        for fileName in os.listdir(inputdir):
            if fileName.endswith(extension):
                files.append((os.path.join(inputdir, fileName), clase))
    np.random.shuffle(files)
    if tam is not None:
        files = files[:int(len(files)*tam)]
    r = []
    last=-1
    print("Se van a cargar "+str(len(files))+" imagenes.")
    for i,(fileName,clase) in enumerate(files):
        if int((float(i)/len(files))*100.)!=last:
            last=int((float(i)/len(files))*100.)
            if last%10==0:
                print(str(last)+"% Completado")
        r.append((np.array(misc.imread(fileName)), clase))
    return (np.array([img for img,_ in r]), np.array([int(c)-32 for _,c in r]))

(X_train, y_train) = get_images('dataset/train', 0.02)
(X_valid, y_valid) = get_images('dataset/valid', 0.1)
(X_test, y_test) = get_images('dataset/test', 0.1)

if K.image_dim_ordering() == 'th':
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_valid = X_valid.reshape(X_valid.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_valid = X_valid.reshape(X_valid.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

X_train = X_train.astype('float32')
X_valid = X_valid.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_valid /= 255
X_test /= 255
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_valid.shape[0], 'valid samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_valid = np_utils.to_categorical(y_valid, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

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

model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          verbose=1, validation_data=(X_valid, Y_valid))

#model.save('model'+str(int(time()))+'.h5')

score = model.evaluate(X_train, Y_train, verbose=0)
print('Train score:', score[0])
print('Train accuracy:', score[1])  
score = model.evaluate(X_valid, Y_valid, verbose=0)
print('Valid score:', score[0])
print('Valid accuracy:', score[1])
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

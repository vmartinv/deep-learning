'''Trains a simple deep NN on the MNIST dataset.

Gets to 98.40% test accuracy after 20 epochs
(there is *a lot* of margin for parameter tuning).
2 seconds per epoch on a K520 GPU.
'''

from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot

def prepare_show(face):
    m = face.min()
    M = face.max()
    return (face-m)/(M-m)

batch_size = 128
nb_classes = 10
nb_epoch = 5

# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# redimensionar a [samples][pixels][ancho][alto]
X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1], X_train.shape[2])
X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1], X_test.shape[2])

print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

# convert from int to float
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

# generador de batches, con tratamiento de los datos
datagen = ImageDataGenerator(
    # pone la media de todo el dataset en 0:
    featurewise_center=False,
    # pone la media de cada muestra en 0:
    samplewise_center=False, 
    # pone la varianza de todo el dataset en 1:
    featurewise_std_normalization=False,
    # pone la varianza de cada muestra en 1:
    samplewise_std_normalization=False,  
    # aplica ZCA whitening:
    zca_whitening=False,
    # rota aleatoriamente las imagenes en el rango (en grados)
    rotation_range=0,
    # mueve horizonalmente las imagenes aleatoriamente: (fraccion del ancho)
    width_shift_range=0.1, 
    # mueve verticalmente las imagenes aleatoriamente: (fraccion del alto)
    height_shift_range=0.1,
    # voltea horizonalmente las imagenes aleatoriamente:
    horizontal_flip=False,
    # voltea verticalmente las imagenes aleatoriamente:
    vertical_flip=False)
    
# proveemos los datos de entrenamiento al generados
datagen.fit(X_train)
    
# genera una vista previa de las imagenes procesadas
for X_batch, y_batch in datagen.flow(X_train, y_train, batch_size=9):
    # crea una grilla con los 10 digitos
    for i in range(0, 9):
        pyplot.subplot(330 + 1 + i)
        pyplot.imshow(prepare_show(X_batch[i].reshape(28, 28)),
                      cmap=pyplot.get_cmap('gray'))
    # muestra el plot
    pyplot.show()
    break
    
model = Sequential()
#En el primer paso aplanamos los datos
model.add(Flatten(input_shape=X_train.shape[1:]))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(10))
model.add(Activation('softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

history = model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size),
                    samples_per_epoch=X_train.shape[0], nb_epoch=nb_epoch,
                        validation_data=(X_test, Y_test))

score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])


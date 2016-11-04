import base
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from top_k_metric import top3
import numpy as np

# number of convolutional filters to use
nb_filters = 64
nb_filters2 = 128
nb_filters3 = 256
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (3, 3)
kernel_size2 = (2, 2)

print("Armando red...")
model = Sequential()
model.add(Dropout(0.2, input_shape=base.input_shape))
model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                        border_mode='valid'))
model.add(Activation('relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
model.add(Activation('relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(0.2))


model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))  
model.add(Activation('relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))  
model.add(Activation('relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(0.2))

model.add(Convolution2D(nb_filters2, kernel_size[0], kernel_size[1]))  
model.add(Activation('relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(nb_filters2, kernel_size[0], kernel_size[1]))  
model.add(Activation('relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(0.2))

model.add(Convolution2D(nb_filters3, kernel_size2[0], kernel_size2[1]))  
model.add(Activation('relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(nb_filters3, kernel_size2[0], kernel_size2[1]))  
model.add(Activation('relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(nb_filters3, kernel_size2[0], kernel_size2[1]))  
model.add(Activation('relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(0.2))


model.add(Flatten())
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(base.nb_classes))
model.add(Activation('softmax'))


print("Compilando...")
model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy', top3 ])
model.summary()

trainer = base.Trainer('red_martin', train_data=base.dataset("dataseth5-featMean/train.h5", "Train"),
                                    valid_data=base.dataset("dataseth5-featMean/valid.h5", "Valid"),
                                    test_data=base.dataset("dataseth5-featMean/test.h5", "Test"))

trainer.train(model, nb_epoch=100, samples_per_epoch=269018) #usa todo el dataset
#~ trainer.train(model, nb_epoch=3, samples_per_epoch=128) #usa todo el dataset
trainer.save_last_train_history()


trainer.evaluate(model)

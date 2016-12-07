import base
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from top_k_metric import top3
from sklearn.metrics import confusion_matrix
import numpy as np

# number of convolutional filters to use
nb_filters = 32
nb_filters2 = 32
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (3, 3)


#~ not_importants=['^', '`', 'U']
#~ not_importants=set(map(lambda x:ord(x)-32, not_importants))
#~ trainer.train_data.filter(lambda _,y: base.to_categorical(y) not in not_importants)

#~ trainer = base.LazyTrainer('red_mariano')
#~ trainer.train_data.preview()

print("Armando red...")
model = Sequential()

#~ model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                        #~ border_mode='valid',
                        #~ input_shape=input_shape))
#~ model.add(Activation('relu'))
#~ model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
#~ model.add(Activation('relu'))
#~ model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
#~ model.add(Activation('relu'))
#~ model.add(MaxPooling2D(pool_size=pool_size))
#~ model.add(Dropout(0.5))


model.add(Convolution2D(nb_filters2, kernel_size[0], kernel_size[1],
                        border_mode='valid',
                        input_shape=base.input_shape))
model.add(Activation('relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(nb_filters2, kernel_size[0], kernel_size[1]))
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

model.add(Convolution2D(nb_filters2, kernel_size[0], kernel_size[1]))  
model.add(Activation('relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(nb_filters2, kernel_size[0], kernel_size[1]))  
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
model.add(Convolution2D(nb_filters2, kernel_size[0], kernel_size[1]))  
model.add(Activation('relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(0.2))

#~ No Estaba
#~ model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))  
#~ model.add(Activation('relu'))
#~ model.add(ZeroPadding2D((1, 1)))
#~ model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))  
#~ model.add(Activation('relu'))
#~ model.add(ZeroPadding2D((1, 1)))
#~ model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))  
#~ model.add(Activation('relu'))
#~ model.add(ZeroPadding2D((1, 1)))
#~ model.add(MaxPooling2D(pool_size=pool_size))
#~ model.add(Dropout(0.2))


model.add(Flatten())
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.5))
#model.add(Dense(128))
#model.add(Activation('relu'))
#model.add(Dropout(0.5))
model.add(Dense(base.nb_classes))
model.add(Activation('softmax'))


#model = Sequential()
#~ model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                        #~ border_mode='valid',
                        #~ input_shape=base.input_shape))
#~ model.add(MaxPooling2D(pool_size=(2, 2)))
#~ model.add(Convolution2D(15, 3, 3, activation='relu'))
#~ model.add(MaxPooling2D(pool_size=(2, 2)))
#~ model.add(Dropout(0.2))
#~ model.add(Flatten())
#~ model.add(Dense(128, activation='relu'))
#~ model.add(Dense(base.nb_classes))
#~ model.add(Dense(base.nb_classes, activation='softmax'))

   

        
print("Compilando...")
model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy', top3 ])
model.summary()
              
#~ model.compile(loss='categorical_crossentropy',
              #~ optimizer='adadelta',
              #~ metrics=['accuracy' ])
        

#~ trainer = base.Trainer('redMarianoPro', train_data=base.dataset("dataset/train", "Train"),
                                    #~ valid_data=base.dataset("dataset/valid", "Valid"),
                                    #~ test_data=base.dataset("dataset/test", "Test"))
trainer = base.Trainer('redMarianoPro', train_data=base.dataset("dataseth5/train.h5", "Train"),
                                    valid_data=base.dataset("dataseth5/valid.h5", "Valid"),
                                    test_data=base.dataset("dataseth5/test.h5", "Test"))
#~ trainer.train(model, nb_epoch=2, samples_per_epoch=10240, nb_val_samples=5000) 
trainer.train(model, nb_epoch=100, samples_per_epoch=269018) #usa todo el dataset
#~ trainer.train(model, nb_epoch=12, samples_per_epoch=269018) #usa todo el dataset
#~ trainer.train(model, nb_epoch=3, samples_per_epoch=100) #usa todo el dataset

#~ model = load_model('redMarianoPro--01-Nov-2016--10-40--model.h5')

trainer.evaluate(model)

#~ valid_data=base.H5Dataset("dataseth5/valid.h5", "Valid")
#~ valid_data.evaluate(model)








#~ badmeasure = np.arange(91)


import base
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D

LOAD_MODEL = False

# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (3, 3)

trainer = base.LazyTrainer('red_mariano')
#~ trainer.train_data.preview()

if LOAD_MODEL:
    model = load_model("red_orig.py-model-1477598524.h5")
else:
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
    
    
    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                            border_mode='valid',
                            input_shape=base.input_shape))
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
    
    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))  
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
    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))  
    model.add(Activation('relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(0.2))
    
    #~ model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))  
    #~ model.add(Activation('relu'))
    #~ model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))  
    #~ model.add(Activation('relu'))
    #~ model.add(ZeroPadding2D((1, 1)))
    #~ model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))  
    #~ model.add(Activation('relu'))
    #~ model.add(ZeroPadding2D((1, 1)))
    #~ model.add(MaxPooling2D(pool_size=pool_size))
    #~ model.add(Dropout(0.2))


    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
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
                  metrics=['accuracy'])

    #~ trainer.train(model, nb_epoch=2, samples_per_epoch=10240, nb_val_samples=5000) 
    trainer.train(model, nb_epoch=12, samples_per_epoch=269018, nb_val_samples=25000) #usa todo el dataset
    trainer.save_last_train_history()

trainer.evaluate(model)

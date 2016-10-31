import base
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D, Merge
from keras.preprocessing.image import DirectoryIterator
from keras import backend as K

class ImageDataGeneratorWrapper(base.ExtensibleImageDataGenerator):
    def flow_from_directory(self, directory,
                                target_size=(256, 256), color_mode='rgb',
                                classes=None, class_mode='categorical',
                                batch_size=32, shuffle=True, seed=None,
                                save_to_dir=None, save_prefix='', save_format='jpeg'):
            return DirectoryIteratorWrapper(
                directory, self,
                target_size=target_size, color_mode=color_mode,
                classes=classes, class_mode=class_mode,
                dim_ordering=self.dim_ordering,
                batch_size=batch_size, shuffle=shuffle, seed=seed,
                save_to_dir=save_to_dir, save_prefix=save_prefix, save_format=save_format)

class DirectoryIteratorWrapper(DirectoryIterator):
    def next(self):
        batch_x, batch_y=super(DirectoryIteratorWrapper, self).next()
        return [batch_x, batch_x, batch_x], batch_y
        
imgDataGen = ImageDataGeneratorWrapper(featurewise_center=False,
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
    dim_ordering=K.image_dim_ordering()).add(lambda x:1-x)
    

# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (4, 2)

trainer = base.LazyTrainer('red_mariano2', train_data=base.LazyDataset("dataset/train", "Train", imgDataGen)
                                         , valid_data=base.LazyDataset("dataset/valid", "Valid", imgDataGen)
                                         , test_data=base.LazyDataset("dataset/test", "Test", imgDataGen))
#~ trainer.train_data.preview()

def miModelo(kernel_size):
    model = Sequential()
     
    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                            border_mode='valid',
                            input_shape=base.input_shape))
    model.add(Activation('relu'))
    model.add(ZeroPadding2D((kernel_size[0]-1, kernel_size[1]-1)))
    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
    model.add(Activation('relu'))
    model.add(ZeroPadding2D((kernel_size[0]-1, kernel_size[1]-1)))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(0.2))
    model.add(Flatten())
    return model

    
    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))  
    model.add(Activation('relu'))
    model.add(ZeroPadding2D((kernel_size[0]-1, kernel_size[1]-1)))
    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))  
    model.add(Activation('relu'))
    model.add(ZeroPadding2D((kernel_size[0]-1, kernel_size[1]-1)))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(0.2))
    
    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))  
    model.add(Activation('relu'))
    model.add(ZeroPadding2D((kernel_size[0]-1, kernel_size[1]-1)))
    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))  
    model.add(Activation('relu'))
    model.add(ZeroPadding2D((kernel_size[0]-1, kernel_size[1]-1)))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(0.2))
    
    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))  
    model.add(Activation('relu'))
    model.add(ZeroPadding2D((kernel_size[0]-1, kernel_size[1]-1)))
    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))  
    model.add(Activation('relu'))
    model.add(ZeroPadding2D((kernel_size[0]-1, kernel_size[1]-1)))
    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))  
    model.add(Activation('relu'))
    model.add(ZeroPadding2D((kernel_size[0]-1, kernel_size[1]-1)))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(0.2))
    
    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))  
    model.add(Activation('relu'))
    model.add(ZeroPadding2D((kernel_size[0]-1, kernel_size[1]-1)))
    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))  
    model.add(Activation('relu'))
    model.add(ZeroPadding2D((kernel_size[0]-1, kernel_size[1]-1)))
    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))  
    model.add(Activation('relu'))
    model.add(ZeroPadding2D((kernel_size[0]-1, kernel_size[1]-1)))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(0.2))

    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))  
    model.add(Activation('relu'))
    model.add(ZeroPadding2D((kernel_size[0]-1, kernel_size[1]-1)))
    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))  
    model.add(Activation('relu'))
    model.add(ZeroPadding2D((kernel_size[0]-1, kernel_size[1]-1)))
    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))  
    model.add(Activation('relu'))
    model.add(ZeroPadding2D((kernel_size[0]-1, kernel_size[1]-1)))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(0.2))
    model.add(Flatten())
    return model
    
print("Armando red...")
model = Sequential()


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
merged = Merge([miModelo((4,2)), miModelo((2,4)), miModelo((3,3))], mode='concat')

model = Sequential()
model.add(merged)

model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(base.nb_classes))
model.add(Activation('softmax'))


#model = Sequential()
#~ model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                        #~ border_mode='valid',
                        #~ input_shape=input_shape))
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
trainer.train(model, nb_epoch=50, samples_per_epoch=269018, nb_val_samples=25000) #usa todo el dataset
trainer.save_last_train_history()

trainer.evaluate(model)

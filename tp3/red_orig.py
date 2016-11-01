import base
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D

# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (3, 3)


#~ trainer = base.Trainer('red_orig', train_data=base.dataset("dataset/train", "Train"),
                                    #~ valid_data=base.dataset("dataset/valid", "Valid"),
                                    #~ test_data=base2.dataset("dataset/test", "Test"))
trainer = base.Trainer('red_orig', train_data=base.dataset("dataseth5/train.h5", "Train"),
                                    valid_data=base.dataset("dataseth5/valid.h5", "Valid"),
                                    test_data=base.dataset("dataseth5/test.h5", "Test"))
trainer.train_data.preview()

print("Armando red...")
model = Sequential()

model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                        border_mode='valid',
                        input_shape=base.input_shape))
model.add(Activation('relu'))
model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(base.nb_classes))
model.add(Activation('softmax'))

print("Compilando...")
model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

#~ trainer.train(model, nb_epoch=2, samples_per_epoch=10240, nb_val_samples=5000) 
trainer.train(model, nb_epoch=12, samples_per_epoch=269018, nb_val_samples=25000) #usa todo el dataset
trainer.save_last_train_history()

trainer.evaluate(model)

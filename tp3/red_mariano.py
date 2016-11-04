import base
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from top_k_metric import top_k_categorical_accuracy
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


def top3(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred,3)
    

        
print("Compilando...")
model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy', top3 ])
model.summary()
              
#~ model.compile(loss='categorical_crossentropy',
              #~ optimizer='adadelta',
              #~ metrics=['accuracy' ])
        
EPOCAS = 100

#~ trainer = base.Trainer('redMarianoPro', train_data=base.dataset("dataset/train", "Train"),
                                    #~ valid_data=base.dataset("dataset/valid", "Valid"),
                                    #~ test_data=base.dataset("dataset/test", "Test"))
trainer = base.Trainer('redMarianoPro2', train_data=base.dataset("dataseth5/train.h5", "Train"),
                                    valid_data=base.dataset("dataseth5/valid.h5", "Valid"),
                                    test_data=base.dataset("dataseth5/test.h5", "Test"))
#~ trainer.train(model, nb_epoch=2, samples_per_epoch=10240, nb_val_samples=5000) 
trainer.train(model, nb_epoch=12, samples_per_epoch=269018) #usa todo el dataset
#~ trainer.train(model, nb_epoch=12, samples_per_epoch=269018) #usa todo el dataset
#~ trainer.train(model, nb_epoch=3, samples_per_epoch=100) #usa todo el dataset
trainer.save_last_train_history()
trainer = base.Trainer('redMarianoPro3', train_data=base.dataset("dataseth5/train.h5", "Train"),
                                    valid_data=base.dataset("dataseth5/valid.h5", "Valid"),
                                    test_data=base.dataset("dataseth5/test.h5", "Test"))
trainer.train(model, nb_epoch=EPOCAS-12, samples_per_epoch=269018) #usa todo el dataset
trainer.save_last_train_history()

#~ model = load_model('redMarianoPro--01-Nov-2016--10-40--model.h5')

trainer.evaluate(model)

#~ valid_data=base.H5Dataset("dataseth5/valid.h5", "Valid")
#~ valid_data.evaluate(model)

test_data=base.H5Dataset("dataseth5/test.h5", "Test")
X, Y = test_data.get_XY()
#~ X = X[:50000]
#~ Y = Y[:50000]
Yclases = []
for yi in Y:
	Yclases.append(np.argmax(yi))

Ypredict = model.predict(X)
YPredictclases = []
for yi in Ypredict:
	YPredictclases.append(np.argmax(yi))

#~ print(Yclases)
#~ print(YPredictclases)


#~ Yclases = [0,0,0,1,1,1,2,2,2]
#~ YPredictclases = [1,1,2, 0,0,1, 2,2,1]

CM = confusion_matrix(Yclases,YPredictclases)


CM2 = []
for i in range(0,len(CM)):
	CM2.append( list( map( lambda x : x / float(sum(CM[i])), CM[i]) ) ) 
total = 269018
cantidades = [41275, 105, 598, 51, 0, 0, 15, 741, 95, 93, 8, 2, 2612, 770, 2970, 16, 245, 237, 77, 98, 67, 79, 65, 29, 75, 97, 91, 84, 0, 0, 0, 63, 0, 639, 449, 429, 255, 318, 285, 365, 452, 687, 76, 104, 315, 761, 368, 191, 453, 7, 334, 486, 767, 119, 90, 413, 7, 69, 4, 0, 0, 0, 0, 0, 0, 17241, 3272, 5796, 8078, 27347, 4642, 3962, 11608, 14989, 216, 1265, 8665, 5174, 15241, 15733, 4068, 176, 13660, 13418, 19458, 5555, 2165, 3858, 341, 3894, 95]
tolerancia = 0.1
for i in range(len(CM)):
	actual = []
	for j in range(len(CM[i])):
		if i != j and CM[i][j]> tolerancia*sum(CM[i]):
			print( "%s %s %.2f" %(chr(i+32), chr(j+32), CM[i][j]/float(sum(CM[i])) ))


#~ badmeasure = np.arange(91)


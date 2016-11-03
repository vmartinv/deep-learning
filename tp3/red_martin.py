import base
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten, Lambda
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from top_k_metric import top_k_categorical_accuracy
from sklearn.metrics import confusion_matrix
from keras import backend as K
import numpy as np

# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (3, 3)

#~ trainer = base.Trainer('redMarianoPro', train_data=base.dataset("dataset/train", "Train"),
                                    #~ valid_data=base.dataset("dataset/valid", "Valid"),
                                    #~ test_data=base.dataset("dataset/test", "Test"))
trainer = base.Trainer('red_martin', train_data=base.dataset("dataseth5/train.h5", "Train"),
                                    valid_data=base.dataset("dataseth5/valid.h5", "Valid"),
                                    test_data=base.dataset("dataseth5/test.h5", "Test"))

info=[('Espacios', 13027), ('e', 8197), ('t', 6022), ('a', 5327), ('o', 5061), ('i', 4502), ('n', 4480), ('s', 4219), ('r', 3950), ('h', 3939), ('d', 2659), ('l', 2632), ('u', 1790), ('c', 1679), ('f', 1636), ('m', 1533), ('w', 1432), ('g', 1260), ('y', 1174), ('p', 1138), ('b', 988), (',', 827), ('.', 685), ('v', 651), ('k', 372), ('T', 266), ('-', 224), ('A', 176), ('S', 173), ('"', 149), ('x', 146), ('I', 146), ('B', 120), ("'", 108), ('C', 105), ('H', 101), ('1', 98), ('M', 95), ('G', 91), ('L', 76), ('P', 73), ('q', 70), ('E', 70), ('W', 60), ('R', 60), ('(', 60), (')', 59), ('F', 56), ('j', 55), (';', 52), ('0', 47), ('z', 40), ('D', 37), ('2', 36), ('N', 34), ('O', 33), ('J', 33), (':', 26), ('9', 25), ('3', 24), ('V', 23), ('5', 19), ('K', 17), ('6', 16), ('4', 16), ('8', 15), ('!', 13), ('?', 11), ('7', 11), ('Y', 10), ('U', 7), ('#', 4), ('Q', 2), ('Z', 1), ('*', 1), ('&', 1)]
info[0]=(' ', info[0][1])
#~ TOT_CLASS = 80
#~ importants=set(map(lambda x:ord(x[0])-32, info[:TOT_CLASS]))
#~ trainer.train_data.filter(lambda _,y: base.to_categorical(y) in importants)
#~ not_importants=set(map(lambda x:ord(x)-32, not_importants))
#~ trainer.train_data.filter(lambda _,y: base.to_categorical(y) not in not_importants)

not_importants=['^', '`', 'U']
not_importants=set(map(lambda x:ord(x)-32, not_importants))
trainer.train_data.filter(lambda _,y: base.to_categorical(y) not in not_importants)


print("Armando red...")
model = Sequential()

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


model.add(Flatten())
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
#~ model.add(Dense(TOT_CLASS))


#~ def mapper(x):
    #~ M = np.concatenate((np.identity(TOT_CLASS), np.ones((TOT_CLASS, base.nb_classes-TOT_CLASS))), axis=1)
    #~ return K.dot(x, M)

#~ def mapper_output_shape(input_shape):
    #~ shape = list(input_shape)
    #~ assert len(shape) == 2  # only valid for 2D tensors
    #~ shape[-1] = base.nb_classes
    #~ return tuple(shape)

#~ model.add(Lambda(mapper, output_shape=mapper_output_shape))

#~ model.add(Activation('relu'))
model.add(Dense(base.nb_classes))
model.add(Activation('softmax'))


def top3(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred,3)
    

        
print("Compilando...")
model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy', top3 ])
              
#~ model.compile(loss='categorical_crossentropy',
              #~ optimizer='adadelta',
              #~ metrics=['accuracy' ])
              

trainer.train(model, nb_epoch=100, samples_per_epoch=268928) #usa todo el dataset
#~ trainer.train(model, nb_epoch=3, samples_per_epoch=128) #usa todo el dataset
#~ trainer.save_last_train_history()

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


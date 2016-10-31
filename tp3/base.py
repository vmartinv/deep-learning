from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility
import os
from keras.utils import np_utils
from keras import backend as K
from scipy import misc
from time import time, strftime, localtime
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from shutil import rmtree
import h5py

nb_classes = 91
# input image dimensions
img_rows, img_cols = 32, 32
input_shape = (1, img_rows, img_cols) if K.image_dim_ordering() == 'th' else (img_rows, img_cols, 1)  

''' Esta clase permite agregarle nuevos procesamientos al DataGenerator
    Ejemplo de generador que invierte los colores:
    datagen = ExtensibleImageDataGenerator(
                rescale=1/255
              ).add(lambda x: 1-x)
    Nota: Los add se pueden 'apilar'
'''
class ExtensibleImageDataGenerator(ImageDataGenerator):
    def __init__(self, **kwargs):
        super(ExtensibleImageDataGenerator, self).__init__(**kwargs)
        self.custom_processings = lambda x:x

    def standardize(self, x):
        x = super(ExtensibleImageDataGenerator, self).standardize(x)
        return self.custom_processings(x)

    def add(self, g):
        custom_processings = self.custom_processings
        self.custom_processings = lambda x: g(custom_processings(x))
        return self

default_datagen = ExtensibleImageDataGenerator(
    featurewise_center=False,
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
    dim_ordering=K.image_dim_ordering(),
    ).add(lambda x: 1-x)

def to_categorical(y):#to_categorical([0 0 1 0 0]) = 2, porque el 1 esta en la posicion 2
    return np.nonzero(y)[0][0]

class LazyDataset(object):
    def __init__(self, directory, name=None, datagen=default_datagen, batch_size=128, **kwargs):
        self.name = name if name else directory
        self.gen_gen = lambda: (print('Cargando %s...'%(self.name)), datagen.flow_from_directory(directory=directory, target_size=(img_rows, img_cols), color_mode='grayscale', batch_size=batch_size, **kwargs))[1]
        self.gen = None
        
    def get_data(self):
        if self.gen is None: self.gen = self.gen_gen()
        return self.gen
    
    def evaluate(self, model, val_samples=50000, **kwargs):
        score = model.evaluate_generator(self.get_data(), val_samples=val_samples, **kwargs)
        print(self.name+' loss:', score[0])
        print(self.name+' accuracy:', score[1])
    
    # genera una vista previa de las imagenes procesadas
    def preview(self, directory="preview"):
        def prepare_show(face):
            m, M = face.min(), face.max()
            return (face-m)/(M-m)
        rmtree(directory, ignore_errors=True)
        os.makedirs(directory)
        X_batch, y_batch = self.get_data().next()
        for i,(img, y) in enumerate(zip(X_batch, y_batch)):
            misc.imsave(os.path.join(directory, str(to_categorical(y)).zfill(3)+'-'+str(i)+'.png'), prepare_show(img.reshape(img_rows, img_cols)))

class NameGen(object):
    def __init__(self, base_name):
        self.name = base_name + '--' + strftime("%d-%b-%Y--%H-%M", localtime())
        
    def getname(self):
        return self.name
        
    def getfile(self, suffix):
        return self.name + '--' + suffix

class BaseTrainer(object):
    def __init__(self, name, train_data, valid_data, test_data):
        self.namegen = NameGen(name)
        self.train_data, self.valid_data, self.test_data = train_data, valid_data, test_data    

    def save_model(self, model):
        file_name = self.namegen.getfile('model.h5')
        print("Guardando pesos en "+file_name+"...")
        model.save(file_name)
        
    def save_last_train_history(self):
        print("Guardando historial...")
        # summarize history for accuracy
        plt.plot(self.history.history['acc'], 'bo-')
        plt.plot(self.history.history['val_acc'], 'go-')
        plt.title('')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.grid(True)
        plt.legend(['Training set', 'Validation set'], loc='lower right')
        plt.savefig(self.namegen.getfile('acc.png'), bbox_inches='tight', dpi = 300)
        plt.clf()
        # summarize history for loss
        plt.plot(self.history.history['loss'], 'bo-')
        plt.plot(self.history.history['val_loss'], 'go-')
        plt.title('')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.grid(True)
        plt.legend(['Training set', 'Validation set'], loc='upper right')
        plt.savefig(self.namegen.getfile('loss.png'), bbox_inches='tight', dpi = 300)
        plt.clf()
        

class LazyTrainer(BaseTrainer):
    def __init__(self, name, train_data=LazyDataset("dataset/train", "Train"), valid_data=LazyDataset("dataset/valid", "Valid"), test_data=LazyDataset("dataset/test", "Test")):
        super(LazyTrainer, self).__init__(name, train_data, valid_data, test_data)

    def train(self, model, samples_per_epoch=269018, nb_epoch=12, verbose=1, nb_val_samples=25000, **kwargs):
        print("Entrenando red: "+self.namegen.getname())
        self.history = model.fit_generator(self.train_data.get_data(), samples_per_epoch=samples_per_epoch, nb_epoch=nb_epoch,
                  verbose=verbose, validation_data=self.valid_data.get_data(), nb_val_samples=nb_val_samples, **kwargs)
        self.save_model(model)

    def evaluate(self, model):
        print("Evaluando modelo...")
        self.train_data.evaluate(model)
        self.valid_data.evaluate(model)
        self.test_data.evaluate(model)

class H5Dataset(object):
    def __init__(self, h5file, name=None, **kwargs):
        self.name = name if name else h5file
        self.h5file = h5file
        self.X = self.Y = None
    
    def load_data(self):
        print("Cargando %s..." % (self.name) )
        with h5py.File(self.h5file,'r') as hf:
            self.X = np.array(hf.get('X'))
            self.Y = np.array(hf.get('Y'))
            print("Found %d images belonging to %d classes." % (self.Y.shape[0], self.Y.shape[1]))
    
    def get_data(self):
        if self.X is None: self.load_data()
        return self.X, self.Y
        

    def evaluate(self, model, verbose=0, **kwargs):
        X, Y = self.get_data()
        if X.shape[0]:
            score = model.evaluate(X, Y, verbose=verbose, **kwargs)
            print(self.name+' loss:', score[0])
            print(self.name+' accuracy:', score[1])
            return score
        else:
            print("No hay data!")
        return None

    def filter(self, f):
        X, Y = self.get_data()
        self.X = []
        self.Y = []
        for x,y in zip(X,Y):
            if f(x, y):
                self.X.append(x)
                self.Y.append(y)
        self.X = np.array(self.X)
        self.Y = np.array(self.Y)
        print("Se filtraron %d imagenes." % (self.X.shape[0]))
            
    # genera una vista previa de las imagenes procesadas
    def preview(self, directory="preview"):
        def prepare_show(face):
            m, M = face.min(), face.max()
            return (face-m)/(M-m)
        rmtree(directory, ignore_errors=True)
        os.makedirs(directory)
        X, Y = self.get_data()
        X_batch, y_batch = X[:128], Y[:128]
        for i,(img, y) in enumerate(zip(X_batch, y_batch)):
            misc.imsave(os.path.join(directory, str(to_categorical(y)).zfill(3)+'-'+str(i)+'.png'), prepare_show(img.reshape(img_rows, img_cols)))
            
        
class H5Trainer(BaseTrainer):
    def __init__(self, name, train_data=H5Dataset("dataseth5/train.h5", "Train"), valid_data=H5Dataset("dataseth5/valid.h5", "Valid"), test_data=H5Dataset("dataseth5/test.h5", "Test")):
        super(H5Trainer, self).__init__(name, train_data, valid_data, test_data)

    def train(self, model, nb_epoch=12, samples_per_epoch=269018, verbose=1, batch_size=128, **kwargs):
        print("Entrenando red: "+self.namegen.getname())
        X_train, Y_train = self.train_data.get_data()
        X_valid, Y_valid = self.valid_data.get_data()
        X_train, Y_train = X_train[:samples_per_epoch], Y_train[:samples_per_epoch]  #VERY DIRTY!! FIX!
        self.history = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
                  verbose=verbose, validation_data=(X_valid, Y_valid), **kwargs)
        self.save_model(model)

    def evaluate(self, model):
        print("Evaluando modelo...")
        self.train_data.evaluate(model)
        self.valid_data.evaluate(model)
        self.test_data.evaluate(model)

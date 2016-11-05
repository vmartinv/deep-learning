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
from keras.callbacks import ModelCheckpoint
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

class BaseDataset(object):
    def __init__(self, name):
        self.name = name
    
    def evaluate(self, model, val_samples=50000, **kwargs):
        gen = self.get_gen()
        if gen is not None:
            score = model.evaluate_generator(gen, val_samples=val_samples, **kwargs)
            for i in range(len(model.metrics_names)):
                print('{} {}: {:2f}'.format(self.name, model.metrics_names[i], score[i]))
        else:
            print("No hay data!")
    
    # genera una vista previa de las imagenes procesadas
    def preview(self, directory="preview"):
        def prepare_show(face):
            m, M = face.min(), face.max()
            return (face-m)/(M-m)
        if os.path.exists(directory):
            rmtree(directory)
        os.makedirs(directory)
        X_batch, y_batch = self.get_gen().next()
        for i,(img, y) in enumerate(zip(X_batch, y_batch)):
            misc.imsave(os.path.join(directory, str(to_categorical(y)).zfill(3)+'-'+str(i)+'.png'), prepare_show(img.reshape(img_rows, img_cols)))

class LazyDataset(BaseDataset):
    def __init__(self, directory, name=None, datagen=None, batch_size=128, **kwargs):
        if datagen is None: datagen=default_datagen
        super(LazyDataset, self).__init__(name if name else directory)
        self.gen_gen = lambda: (print('Cargando {}...'.format(self.name)), datagen.flow_from_directory(directory=directory, target_size=(img_rows, img_cols), color_mode='grayscale', batch_size=batch_size, **kwargs))[1]
        self.gen = None
        
    def get_XY(self):
        return self.get_gen().next()
        
    def get_gen(self):
        if self.gen is None: self.gen = self.gen_gen()
        return self.gen
        
class H5Dataset(BaseDataset):
    def __init__(self, h5file, name=None, datagen=None, batch_size=128, **kwargs):
        super(H5Dataset, self).__init__(name if name else h5file)
        self.h5file = h5file
        self.batch_size = batch_size
        self.X = self.Y = None
        self.datagen = datagen
        self.load_data()
    
    def load_data(self):
        print("Cargando {}...".format(self.name))
        with h5py.File(self.h5file,'r') as hf:
            self.X = np.array(hf.get('X'))
            self.Y = np.array(hf.get('Y'))
            print("Found {} images belonging to {} classes.".format(self.Y.shape[0], self.Y.shape[1]))
            if self.datagen is not None:
                self.datagen.fit(self.X)
    
    def get_XY(self):
        if self.X is None: self.load_data()
        return self.X, self.Y

    def filter(self, f):
        X, Y = self.get_XY()
        self.X = []
        self.Y = []
        for x,y in zip(X,Y):
            if f(x, y):
                self.X.append(x)
                self.Y.append(y)
        self.X = np.array(self.X)
        self.Y = np.array(self.Y)
        print("Se filtraron %d imagenes." % (self.X.shape[0]))
        
    def get_gen(self):
        X, Y = self.get_XY()
        datagen = self.datagen if self.datagen is not None else ImageDataGenerator()
        return datagen.flow(X, Y, batch_size=self.batch_size)

def dataset(source, name=None, datagen=None, batch_size=128, **kwargs):
    if os.path.splitext(source)[1]=='.h5':
        return H5Dataset(source, name, datagen, batch_size, **kwargs)
    else:
        return LazyDataset(source, name, datagen, batch_size, **kwargs)

class NameGen(object):
    def __init__(self, base_name):
        self.name = base_name + '--' + strftime("%d-%b-%Y--%H-%M", localtime())
        
    def get_name(self):
        return self.name
    
    def get_file(self, dire, suffix):
        if not os.path.exists(dire):
            os.makedirs(dire)
        return os.path.join(dire, self.name + '--' + suffix)
        
    def get_model_file(self, suffix):
        return self.get_file("models", suffix)
        
    def get_history_file(self, suffix):
        return self.get_file("histories", suffix)

class Trainer(object):
    def __init__(self, name, train_data, valid_data, test_data):
        self.namegen = NameGen(name)
        self.train_data, self.valid_data, self.test_data = train_data, valid_data, test_data    
    
    def save_model_struct(self, model):
        with open(self.namegen.get_model_file('model-struct.json'), "w") as text_file:
            text_file.write(model.to_json())

    def train(self, model, samples_per_epoch=269018, nb_epoch=12, verbose=1, nb_val_samples=25000, **kwargs):
        print("Entrenando red: "+self.namegen.get_name())
        self.save_model_struct(model)
        checkpointer = ModelCheckpoint(filepath=self.namegen.get_model_file('model-train-weights.h5'), save_weights_only=True, monitor='val_acc', verbose=1, save_best_only=True)
        self.history = model.fit_generator(self.train_data.get_gen(), samples_per_epoch=samples_per_epoch, nb_epoch=nb_epoch,
                  verbose=verbose, validation_data=self.valid_data.get_gen(), nb_val_samples=nb_val_samples, callbacks=[checkpointer], **kwargs)
        self.save_model(model)
        self.save_last_train_history()

    def evaluate(self, model):
        print("Evaluando modelo...")
        self.train_data.evaluate(model)
        self.valid_data.evaluate(model)
        self.test_data.evaluate(model)

    def save_model(self, model):
        restore=False
        if 'top3' in model.model.metrics_names:
            idx=model.model.metrics_names.index('top3')
            metric_name = model.model.metrics_names[idx]
            del model.model.metrics_names[idx]
            metric = model.model.metrics[idx-1]
            del model.model.metrics[idx-1]
            metric_tensor = model.model.metrics_tensors[idx-1]
            del model.model.metrics_tensors[idx-1]
            restore = True
    
        file_name = self.namegen.get_model_file('model.h5')
        print("Guardando pesos en "+file_name+"...")
        self.save_model_struct(model)
        model.save(file_name)
        if restore:
            model.model.metrics_names.insert(idx, metric_name)
            model.model.metrics.insert(idx-1, metric)
            model.model.metrics_tensors.insert(idx-1, metric_tensor)
        
    def save_last_train_history(self):
        print("Guardando historial...")
        # summarize history for accuracy
        plt.plot(self.history.history['acc'], 'bo-')
        plt.plot(self.history.history['val_acc'], 'go-')
        plt.title('')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.grid(True)
        plt.legend(['Conj. Entrenamiento', 'Conj. Validacion'], loc='lower right')
        plt.savefig(self.namegen.get_history_file('acc.png'), bbox_inches='tight', dpi = 300)
        plt.clf()
        # summarize history for loss
        plt.plot(self.history.history['loss'], 'bo-')
        plt.plot(self.history.history['val_loss'], 'go-')
        plt.title('')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.grid(True)
        plt.legend(['Conj. Entrenamiento', 'Conj. Validacion'], loc='upper right')
        plt.savefig(self.namegen.get_history_file('loss.png'), bbox_inches='tight', dpi = 300)
        plt.clf()


from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility
import os
import sys
import multiprocessing
from scipy import misc
from keras.utils import np_utils
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from skimage.filter import threshold_otsu
from skimage.morphology import medial_axis
from PIL import Image
import shutil
import h5py

nb_classes = 91
# input image dimensions
img_rows, img_cols = 32, 32

# the data, shuffled and split between train and test sets
def get_images(base, tam=None):
    print("Cargando "+base+"...")
    extension = '.png'
    files = []
    for clase in os.listdir(base):
        inputdir = os.path.join(base, clase)
        for fileName in os.listdir(inputdir):
            if fileName.endswith(extension):
                files.append((os.path.join(inputdir, fileName), clase))
    np.random.shuffle(files)
    if tam is not None:
        files = files[:int(len(files)*tam)]
    r = []
    last=-1
    print("Se van a cargar "+str(len(files))+" imagenes.")
    for i,(fileName,clase) in enumerate(files):
        if int((float(i)/len(files))*100.) != last:
            last=int((float(i)/len(files))*100.)
            if last%10==0:
                print(str(last)+"% Completado")
        r.append((np.array(misc.imread(fileName)), clase))
    return (np.array([img for img,_ in r]), np.array([int(c)-32 for _,c in r]))

(X_train, Y_train) = get_images('dataset/train', tam = 0.25)
(X_test, Y_test) = get_images('dataset/test')#, tam = 0.01)
(X_valid, Y_valid) = get_images('dataset/valid')#, tam = 0.01)

if K.image_dim_ordering() == 'th':
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    X_valid = X_valid.reshape(X_valid.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    X_valid = X_valid.reshape(X_valid.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_valid = X_valid.astype('float32')
X_train /= 255
X_test /= 255
X_valid /= 255
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_valid.shape[0], 'valid samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(Y_train, nb_classes)
Y_valid = np_utils.to_categorical(Y_valid, nb_classes)
Y_test = np_utils.to_categorical(Y_test, nb_classes)


PREVIEW_DIR = 'preview-orig'
shutil.rmtree(PREVIEW_DIR, ignore_errors=True)
os.makedirs(PREVIEW_DIR)

# genera una vista previa de las imagenes procesadas
def prev(cant=None):
    if not cant:
        cant = X_train.shape[0]
    X_batch = X_train[0:cant]
    if not os.path.exists(PREVIEW_DIR):
        os.makedirs(PREVIEW_DIR)
    for i,img in enumerate(X_batch):
        path = os.path.join(PREVIEW_DIR, str(i)+'.png')
        newImg = img.reshape(img_rows, img_cols)
        misc.imsave(path, newImg)
        
#prev(300)




#H5py
sys.stdout.write("Guardando archivos h5...")
sys.stdout.flush()

H5FOLDER = 'dataseth5-0.25'
if not os.path.exists(H5FOLDER):
    os.makedirs(H5FOLDER)
# Create a new file using defaut properties.
fTrain = h5py.File(H5FOLDER+'/train.h5','w')
fValid = h5py.File(H5FOLDER+'/valid.h5','w')
fTest = h5py.File(H5FOLDER+'/test.h5','w')

# Create a dataset under the Root group.
X_train = fTrain.create_dataset("X",data=X_train)
Y_train = fTrain.create_dataset("Y",data=Y_train)
X_valid = fValid.create_dataset("X",data=X_valid)
Y_valid = fValid.create_dataset("Y",data=Y_valid)
X_test = fTest.create_dataset("X",data=X_test)
Y_test = fTest.create_dataset("Y",data=Y_test)

# Close the file before exiting
fTrain.close()
fValid.close()
fTest.close()

print("OK")

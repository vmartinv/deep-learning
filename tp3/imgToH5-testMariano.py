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

(X_test, Y_test) = get_images('dataset/marianotest')#, tam = 0.01)

if K.image_dim_ordering() == 'th':
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

X_test = X_test.astype('float32')
X_test /= 255
print(X_test.shape[0], 'test samples')


Y_test = np_utils.to_categorical(Y_test, nb_classes)


#H5py
sys.stdout.write("Guardando archivos h5...")
sys.stdout.flush()

H5FOLDER = 'marianoTest'
if not os.path.exists(H5FOLDER):
    os.makedirs(H5FOLDER)
# Create a new file using defaut properties.
fTest = h5py.File(H5FOLDER+'/test.h5','w')

# Create a dataset under the Root group.
X_test = fTest.create_dataset("X",data=X_test)
Y_test = fTest.create_dataset("Y",data=Y_test)

# Close the file before exiting
fTest.close()

print("OK")

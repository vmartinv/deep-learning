from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility
import os
import sys
from scipy import misc
from keras.utils import np_utils
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import shutil
import h5py

nb_classes = 91
# input image dimensions
img_rows, img_cols = 32, 32

def prepare_show(face):
    m = face.min()
    M = face.max()
    return (face-m)/(M-m)

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
        if int((float(i)/len(files))*100.)!=last:
            last=int((float(i)/len(files))*100.)
            if last%10==0:
                print(str(last)+"% Completado")
        r.append((np.array(misc.imread(fileName)), clase))
    return (np.array([img for img,_ in r]), np.array([int(c)-32 for _,c in r]))

(X_train, Y_train) = get_images('dataset/train')#, tam = 0.005)
(X_test, Y_test) = get_images('dataset/test')#, tam = 0.005)
(X_valid, Y_valid) = get_images('dataset/valid')#, tam = 0.005)

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


imgDataGen = ImageDataGenerator(
    featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=True,
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
    rescale=None,
    dim_ordering="th")
#imgGenerator = imgDataGen.flow_from_directory("dataset/train", color_mode='grayscale', target_size=(32,32), seed=1337, batch_size=batch_size, save_to_dir=PREVIEW_DIR)
imgDataGen.fit(X_train)


X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_valid.shape[0], 'valid samples')
print(X_test.shape[0], 'test samples')

PREVIEW_DIR = 'preview'
shutil.rmtree(PREVIEW_DIR, ignore_errors=True)
os.makedirs(PREVIEW_DIR)

# genera una vista previa de las imagenes procesadas
def prev(cant):
    X_batch = X_train[0:cant]
    Y_batch = Y_train[0:cant]
    if not os.path.exists(PREVIEW_DIR):
        os.makedirs(PREVIEW_DIR)
    for i,img in enumerate(X_batch):
        path = os.path.join(PREVIEW_DIR, str(i)+'.png')
        newImg = prepare_show(img.reshape(img_rows, img_cols))
        misc.imsave(path, newImg)
        
prev(200)

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(Y_train, nb_classes)
Y_valid = np_utils.to_categorical(Y_valid, nb_classes)
Y_test = np_utils.to_categorical(Y_test, nb_classes)








#H5py
sys.stdout.write("Guardando archivos h5...")
sys.stdout.flush()

# Create a new file using defaut properties.
fTrain = h5py.File('dataset/train.h5','w')
fValid = h5py.File('dataset/valid.h5','w')
fTest = h5py.File('dataset/test.h5','w')

# Create a dataset under the Root group.
X_train = fTrain.create_dataset("X_train",data=X_train)
Y_train = fTrain.create_dataset("Y_train",data=Y_train)
X_valid = fValid.create_dataset("X_valid",data=X_valid)
Y_valid = fValid.create_dataset("Y_valid",data=Y_valid)
X_test = fTest.create_dataset("X_test",data=X_test)
Y_test = fTest.create_dataset("Y_test",data=Y_test)

# Close the file before exiting
fTrain.close()
fValid.close()
fTest.close()

print("OK")

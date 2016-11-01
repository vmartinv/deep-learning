from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility
import os
import sys
from scipy import misc
from keras.utils import np_utils
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from skimage.filters import threshold_adaptive
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
    zca_whitening=False,
    rotation_range=15,
    width_shift_range=0.,
    height_shift_range=0.,
    shear_range=0.,
    zoom_range=[1,1.15],
    channel_shift_range=0.,
    fill_mode='constant',
    cval=0,
    horizontal_flip=False,
    vertical_flip=False,
    rescale=None,
    dim_ordering="th")

#sys.stdout.write("fitting...")
#sys.stdout.flush()
#imgDataGen.fit(X_train)
#print("OK")




#X_train = X_train.astype('float32')
#X_test = X_test.astype('float32')
#X_valid = X_valid.astype('float32')
#X_train /= 255
#X_test /= 255
#X_valid /= 255
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_valid.shape[0], 'valid samples')
print(X_test.shape[0], 'test samples')

# binarize
X_train = [np.logical_not(threshold_adaptive(image=i[0], block_size=7)) for i in X_train]
X_test = [np.logical_not(threshold_adaptive(image=i[0], block_size=7)) for i in X_test]
X_valid = [np.logical_not(threshold_adaptive(image=i[0], block_size=7)) for i in X_valid]
# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(Y_train, nb_classes)
Y_valid = np_utils.to_categorical(Y_valid, nb_classes)
Y_test = np_utils.to_categorical(Y_test, nb_classes)


#batch = imgDataGen.flow(X_train,Y_train, batch_size=20, save_to_dir=PREVIEW_DIR)


PREVIEW_DIR = 'preview-ft'
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
        
prev(200)











#H5py
sys.stdout.write("Guardando archivos h5...")
sys.stdout.flush()

os.makedirs('dataseth5')
# Create a new file using defaut properties.
fTrain = h5py.File('dataseth5/train2.h5','w')
fValid = h5py.File('dataseth5/valid2.h5','w')
fTest = h5py.File('dataseth5/test2.h5','w')

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

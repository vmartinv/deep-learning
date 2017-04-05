import os
import numpy as np
import cnn
import paths

def list_files(directory, ext='jpg|jpeg|bmp|png'):
    return [os.path.join(directory, f) for f in os.listdir(directory)
            if os.path.isfile(os.path.join(directory, f)) and ("."+ext in f)]


for f in list_files(paths.srcPath(),'png'):
    classes = cnn.applyCNN(f)
    baseName = os.path.basename(f)
    np.save(os.path.join(paths.npyPath(),baseName[0:-4]),classes)

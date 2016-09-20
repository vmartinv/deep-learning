import os, subprocess
from scipy import misc
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt


#Funciones útiles
def get_images(clase, tam=None):
    inputdir = "./"+clase+"/"
    extension = '.jpg'
    r = []
    files = [fileName for fileName in os.listdir(inputdir) if fileName.endswith(extension)]
    np.random.shuffle(files)
    if tam is not None:
        files = files[:min(len(files), tam)]
    return list(map(lambda fileName: misc.imread(os.path.join(os.path.join(inputdir, fileName))), files))
    
def to_gray_scale(face):
    return np.dot(face[...,:3], [0.299, 0.587, 0.114])

def to_float(face):
    face = np.float32(face)
    face /= 255
    return face    

def mean_by_channel(face):
    r = []
    for i in range(face.shape[2]):
        r.append(face[...,i].mean())
    return np.array(r)

def zero_mean(face):
    face -= mean_by_channel(face)
    return face

def var_by_channel(face):
    r = []
    for i in range(face.shape[2]):
        r.append(face[...,i].var())
    return np.array(r)

def one_variance(face):
    variance = var_by_channel(face)
    face /= list(map(sqrt,variance)  )  
    return face

def show(face, title=None):
    m = face.min()
    M = face.max()
    face = (face-m)/(M-m)
    plt.imshow(face)
    if title:
        plt.suptitle(title, fontsize=24, fontweight='bold')
    plt.axis('off')
    plt.show()

def add_depth(face):
    if len(face.shape) < 3:
        x, y = face.shape
        ret = np.empty((x, y, 3), dtype=np.float32)
        ret[:, :, 2] =  ret[:, :, 1] =  ret[:, :, 0] =  face
        return ret
    else:
        return face

def make_squared(face):
    lx, ly, _ = face.shape
    tam = max(lx, ly)
    color = mean_by_channel(face)
    if ly<tam:
        if tam>1+ly:
            face = np.append([[color]*((tam-ly)//2)]*lx, face, axis=1)
        face = np.append(face, [[color]*((tam-ly+1)//2)]*lx, axis=1)
    elif lx<tam:
        if tam>1+lx:
            face = np.append([[color]*ly]*((tam-lx)//2), face, axis=0)
        face = np.append(face, [[color]*ly]*((tam-lx+1)//2), axis=0)
    return face

def resize_image(face,tam):
    return to_float(misc.imresize(face, (tam, tam))) 

def normalize(face):
    face = add_depth(face)
    face = make_squared(to_float(face))
    face = zero_mean(face)
    face = one_variance(face)
    return face






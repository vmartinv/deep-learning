import os, subprocess
from scipy import misc
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt

inputdir = "./borges-words/"
extension = '.png'
final_tam = 28

#Funciones útiles
def get_images():
    r = []
    for fileName in os.listdir(inputdir):
        if fileName.endswith(extension):
            r.append(misc.imread(os.path.join(inputdir, fileName)))
            if len(r)>=10:
                return r
    return r

def to_float(face):
    face = np.float32(face)
    face /= 255
    return face    

def mean_by_channel(face):
    r = []
    for i in range(face.shape[2]):
        r.append(face[...,i].mean())
    return np.array(r)

def var_by_channel(face):
    r = []
    for i in range(face.shape[2]):
        r.append(face[...,i].var())
    return np.array(r)

def show(face):
    #m = face.min()
    #M = face.max()
    #face = (face-m)/(M-m)
    plt.imshow(face/ face.max())    
    plt.show()

#Ejercicio 1
def make_squared(face):
    lx, ly, col = face.shape
    tam = max(lx, ly)
    color = mean_by_channel(face)
    if ly<tam:
        face = np.append([[color]*((tam-ly)//2)]*lx, face, axis=1)
        face = np.append(face, [[color]*((tam-ly+1)//2)]*lx, axis=1)
    elif lx<tam:
        face = np.append([[color]*ly]*((tam-lx)//2), face, axis=0)
        face = np.append(face, [[color]*ly]*((tam-lx+1)//2), axis=0)
    return face

def drop_transparency(face):
    face = np.array([ face[...,0], face[...,1], face[...,2] ])
    face = np.rollaxis(face, 0, 3)
    return face

def show_normalized():
    for face in get_images():
        face = drop_transparency(face)
        X = make_squared(to_float(face))
        X = one_variance(X)
        show(X)   
        #X = zero_mean(X)
        #show(X)   

#Ejercicio 2
def get_slice(face,dx,dy):
    lx, ly, _ = face.shape
    y = np.random.random_integers(0, ly-dy-1)
    x = np.random.random_integers(0, lx-dx-1)
    return face[x:x+dx][:,y:y+dy]

def zero_mean(face):
    face -= mean_by_channel(face)        
    #print("Media por canal: %s " % (mean_by_channel(face)))
    return face

def one_variance(face):
    #print("Varianza por canal: %s " % (var_by_channel(face)))
    variance = var_by_channel(face)
    face /= list(map(sqrt,variance)  )  
    #show(face/ face.max())
    #print("Varianza por canal: %s " % (var_by_channel(face)))
    return face

def normalize(face):
    #print(face)
    #face = zero_mean(face)
    #face = one_variance(face)
    #print(face)
    return face

def f(face):
    show((face*0)+mean_by_channel(face))

def make_slices():
    for face in get_images():
        face = to_float(face)
        #face = get_slice(face, 16, 16)
        #face = zero_mean(face)
        face = one_variance(face)
        #plt.imshow(face)    
        #plt.show()
        show(face)
        #show(f(face))


#show(normalize(to_float(misc.imread('hands.jpg'))))
show_normalized()
#make_slices()
#one_variance(to_float(misc.imread('barco.jpg')))


import os, subprocess
from scipy import misc
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt


#Funciones útiles
def get_images_all():
    inputdir = "./Caltech101/"
    extension = '.jpg'
    r = []
    for clase in os.listdir(inputdir):
        for fileName in os.listdir(os.path.join(inputdir, clase)):
            if fileName.endswith(extension):
                r.append(misc.imread(os.path.join(os.path.join(inputdir, clase), fileName)))
                if len(r)>=10: #Para evitar recorrer todo el dataset
                    return r
    return r

def get_images_borges():
    inputdir = "./borges-words/"
    extension = '.png'
    r = []
    for fileName in os.listdir(inputdir):
        if fileName.endswith(extension):
            r.append(misc.imread(os.path.join(inputdir, fileName)))
            if len(r)>=10: #Para evitar recorrer todo el dataset
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

def show(face):
    m = face.min()
    M = face.max()
    face = (face-m)/(M-m)
    plt.imshow(face) 
    plt.axis('off')
    plt.show()

#Ejercicio 1
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
        face = np.append([[color]*((tam-ly)//2)]*lx, face, axis=1)
        face = np.append(face, [[color]*((tam-ly+1)//2)]*lx, axis=1)
    elif lx<tam:
        face = np.append([[color]*ly]*((tam-lx)//2), face, axis=0)
        face = np.append(face, [[color]*ly]*((tam-lx+1)//2), axis=0)
    return face

# Elimina la cuarta capa para imágenes png que contengan transparencia.
def drop_transparency(face):
    face = np.array([ face[...,0], face[...,1], face[...,2] ])
    face = np.rollaxis(face, 0, 3)
    return face

def resize_image(face,tam):
    return to_float(misc.imresize(face, (tam, tam))) 


def normalize(face):
    face = add_depth(face)
    face = make_squared(to_float(face))
    face = zero_mean(face)
    face = one_variance(face)
    return face


def ejercicio1_main():
    for face in get_images_all():
        face = normalize(face)
        face = resize_image(face,28)
        show(face)   
        
#Ejercicio 2
def get_slice(face,dx,dy):
    lx, ly, _ = face.shape
    y = np.random.random_integers(0, ly-dy-1)
    x = np.random.random_integers(0, lx-dx-1)
    return face[x:x+dx][:,y:y+dy]

def ejercicio2_main():
    for face in get_images_all():
        subface = add_depth(face)
        subface = get_slice(subface,16,16)
        subface = normalize(subface)
        show(subface)


#Ejercicio 4
def ejercicio4_main():
    for face in get_images_borges():
        face = drop_transparency(face)
        face = normalize(face)
        face = resize_image(face,28)
        show(face)

print("Ejercicio 1...")
ejercicio1_main()
print("Ejercicio 2...")
ejercicio2_main()
print("Ejercicio 4...")
ejercicio4_main()




import os, subprocess
from scipy import misc
import numpy as np
import matplotlib.pyplot as plt

inputdir = "./Caltech101/"
extension = '.jpg'
final_tam = 28

def get_images():
    r = []
    for clase in os.listdir(inputdir):
        for fileName in os.listdir(os.path.join(inputdir, clase)):
            if fileName.endswith(extension):
                r.append(misc.imread(os.path.join(os.path.join(inputdir, clase), fileName)))
                if len(r)>=10:
                    return r
    return r

#Ejercicio 1
def show_normalized():
    for face in get_images():
        face = np.float32(face)
        face /= 255
        lx, ly, col = face.shape
        tam = max(lx, ly)
        color = [face[...,0].mean(), face[...,1].mean(), face[...,2].mean()]
        if ly<tam:
        	face = np.append([[color]*((tam-ly)/2)]*lx, face, axis=1)
        	face = np.append(face, [[color]*((tam-ly+1)/2)]*lx, axis=1)
        elif lx<tam:
        	face = np.append([[color]*ly]*((tam-lx)/2), face, axis=0)
        	face = np.append(face, [[color]*ly]*((tam-lx+1)/2), axis=0)
        face = misc.imresize(face, (final_tam, final_tam))
        plt.imshow(face)    
        plt.show()


#Ejercicio 2
def get_slice(imagen,dx,dy):
    lx, ly, _ = imagen.shape
    y = np.random.random_integers(0, ly-dy-1)
    x = np.random.random_integers(0, lx-dx-1)
    x = 0
    y = 0
    print(dx)
    print(dy)
    return imagen[x:x+dx][:,y:y+dy]

def show_slice():
    for face in get_images():
        face = np.float32(face)
        face /= 255
        lx, ly, col = face.shape
        tam = max(lx, ly)
        face = misc.imresize(face, (50, 50))
        print(face.shape)
        face = get_slice(face,16,16)
        print(face.shape)
        plt.imshow(face)    
        plt.show()


show_slice()


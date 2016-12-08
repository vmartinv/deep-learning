import os, subprocess
from scipy import misc
import numpy as np
from utils import *
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from pylab import *
from skimage import color


def show(face, graphfile):
    m = face.min()
    M = face.max()
    face = (face-m)/(M-m)
    fig = plt.imshow(face, cmap=plt.get_cmap('gray'))        
    # muestra el plot
    plt.axis('off')
    plt.savefig(graphfile, bbox_inches = 'tight', pad_inches = 0)
    plt.clf()

def imagePreprocessing( imagen ):
    imagen = color.rgb2gray( imagen )
    return imagen

def stringPreprocessing( toPredict ):
    toPredict = strip_accents(toPredict).encode('ascii','ignore')
    return toPredict

def load_images(pathFiles):
    dataList = []
    for fileName in os.listdir(pathFiles):
        imagen =  misc.imread(os.path.join(pathFiles, fileName))
        imagen =  imagePreprocessing( imagen )
        toPredict = fileName[ fileName.index('-')+2 : len(fileName)-4 ]
        toPredict = stringPreprocessing( toPredict )
        dataList.append( (imagen, toPredict) )
#    for f in textList:
#        print(f)
    return dataList





dataList = load_images("./lines/")

def to_float(face):
    face = np.float32(face)
    face /= 255
    return face 

for img,f in dataList:
    print(f)
    print(img)
    ly, lx = img.shape
    print(img.shape)
    mylist = []
    for y in range(0,ly):
        sumy = 0
        for x in range(0,lx):
            sumy = sumy + img[y][x]
        mylist.append(sumy)
    print(mylist)
    print(ly)
    print(lx)
    print(int(32*lx/ly))
    #img = to_float(misc.imresize(img, (32, lx)))
    img = to_float(misc.imresize(img, (32, int(32*lx/ly) )))
    #img = to_float(misc.imresize(img, (ly, lx)))
    show(img,"caca.png")
    break


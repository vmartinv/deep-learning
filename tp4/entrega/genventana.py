import os
import numpy as np
import preproceso as pp
import sys
import paths
from keras import backend as K

img_rows = 32
img_cols = 32

def windows(nameImg,winWidth,step,finalWidth):
    name,img=nameImg
    wins=[]
    w,h = img.size
    for s in range(0,w-winWidth,step):
        win = img.crop((s,0,s+winWidth,h))
        if winWidth < finalWidth:
            diffL=(finalWidth-winWidth)//2
            diffR=(finalWidth-winWidth+1)//2
            win=pp.fill_border(win,diffL,diffR,img_rows)
        if paths.previewPath():
            if not os.path.exists(os.path.join(paths.previewPath(),'wins', name[0:-4])):
                os.makedirs(os.path.join(paths.previewPath(),'wins', name[0:-4]))
            win.save(os.path.join(paths.previewPath(),'wins', name[0:-4], 'step'+str(s)+'.png'))
        win = np.asarray(win,dtype=np.float32)
        win /= 255
        win -= 0.87370783 #featurewise mean de los datos de tp3
        wins.append(win)
    return (name,reshape(np.array(wins)))

def reshape(imgs):
    if K.image_dim_ordering() == 'th':
        imgs = imgs.reshape(imgs.shape[0], 1, img_rows, img_cols)
    else:
        imgs = imgs.reshape(imgs.shape[0], img_rows, img_cols, 1)
    return imgs

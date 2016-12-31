import os
import re
import os.path
import argparse
import numpy as np
from PIL import Image, ImageFont, ImageOps, ImageEnhance
from PIL.ImageDraw import Draw

def list_files(directory, ext='jpg|jpeg|bmp|png'):
    return [os.path.join(directory, f) for f in os.listdir(directory)
            if os.path.isfile(os.path.join(directory, f)) and ("."+ext in f)]

contrast_ratio = 25

def resize(img,hMax):
    '''
    Define a maximum size. Then, compute a resize ratio by taking min(maxwidth/width, maxheight/height).
    The proper size is oldsize*ratio.
    '''
    (wIm,hIm) = img.size
    ratio = hMax/float(hIm)
    (wNewIm,hNewIm) = (int(wIm*ratio),hMax)
    newIm = img.resize((wNewIm,hNewIm),Image.ANTIALIAS)
    return newIm

def fill_border(img,left,right,hMax):
    arr = np.asarray(img)
    #~ h,w = arr.shape
    #~ col=np.median([arr[0,0],arr[0,w-1],arr[h-1,0],arr[h-1,w-1]])
    #~ col=np.mean(arr)
    col = 255
    #~ print('media-antes: {}'.format(col))
    l=np.full((hMax,left), col, dtype=np.uint8)
    r=np.full((hMax,right), col, dtype=np.uint8)
    arr=np.c_[l,arr,r]
    #~ print('media-dsp: {}\n'.format(np.mean(arr)))
    img=Image.fromarray(arr)
    return img

def crop_and_save_line(image,ymin,ymax,dest_dir,name):
    xmin = 0
    xmax = image.size[0]
    line_image = image.crop((xmin,ymin,xmax,ymax))
    line_image=resize(line_image,32)
    line_image=fill_border(line_image,10,10,32)
    line_image=erase_line(line_image)
    contrast=ImageEnhance.Contrast(line_image)
    line_image=contrast.enhance(contrast_ratio)
    
    if dest_dir:
        line_image.save(os.path.join(dest_dir, name))
    return line_image

def erase_line(img):
    img = np.asarray(img)
    n,m=img.shape
    img=np.transpose(img).tolist()
    for i in range(m):
        for j in range(n):
            if img[i][j]>120:
                img[i][j]=255
    img = np.transpose(np.array(img).astype('uint8'))
    return Image.fromarray(img)
        


def refine_line_bounds(image):
    ymin = 0
    ymax = image.size[1]
    contrast = ImageEnhance.Contrast(image)
    line_np = 255 - np.array(contrast.enhance(2))
    histo = np.square(np.mean(line_np,axis=(1)))
    prob = histo / np.sum(histo)
    y = np.asarray(range(ymin,ymax))
    y_mean = np.dot(prob,y)
    s = (y - y_mean)**2
    s = np.sqrt(np.dot(prob,s))
    ymax = min(ymax,int(round(y_mean+2*s)))
    ymin = max(ymin,int(round(y_mean-1.2*s)))
    return ymin, ymax

def preprocess(source_path,dest_path=None):
    imgs=[]
    if dest_path and not os.path.exists(dest_path):
        os.makedirs(dest_path)
    for f in list_files(source_path,'png'):
        img = Image.open(f)
        img = ImageOps.grayscale(img)
        ymin, ymax = refine_line_bounds(img)
        baseName=os.path.basename(f)
        img = crop_and_save_line(img,ymin,ymax,dest_path, baseName)
        imgs.append((baseName,img))
    return imgs


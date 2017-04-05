import os
import re
import os.path
import argparse
import paths
import numpy as np
from PIL import Image, ImageFont, ImageOps, ImageEnhance
from PIL.ImageDraw import Draw

contrast_ratio = 3

def resize(img,hMax):
    (wIm,hIm) = img.size
    ratio = hMax/float(hIm)
    (wNewIm,hNewIm) = (int(wIm*ratio),hMax)
    newIm = img.resize((wNewIm,hNewIm),Image.ANTIALIAS)
    return newIm

def fill_border(img,left,right,hMax):
    arr = np.asarray(img)
    col = 255
    l=np.full((hMax,left), col, dtype=np.uint8)
    r=np.full((hMax,right), col, dtype=np.uint8)
    arr=np.c_[l,arr,r]
    img=Image.fromarray(arr)
    return img

def crop_and_save_line(image,ymin,ymax,name):
    xmin = 0
    xmax = image.size[0]
    line_image = image.crop((xmin,ymin,xmax,ymax))
    line_image=resize(line_image,32)
    line_image=fill_border(line_image,16,16,32)
    contrast=ImageEnhance.Contrast(line_image)
    line_image=contrast.enhance(contrast_ratio)
    if paths.previewPath():
        line_image.save(os.path.join(paths.previewPath() , name))
    return line_image

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

def preprocessOne(f):
    img = Image.open(f)
    img = ImageOps.grayscale(img)
    ymin, ymax = 0, img.size[1]
    baseName=os.path.basename(f)
    img = crop_and_save_line(img,ymin,ymax, baseName)
    return baseName,img


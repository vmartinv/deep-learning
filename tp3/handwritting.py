import os
import sys

import numpy as np

from PIL.ImageDraw import Draw
from PIL import ImageFont
from PIL import Image

import xml.etree.ElementTree as ET

def list_files(directory, ext='jpg|jpeg|bmp|png'):
    return [os.path.join(directory, f) for f in os.listdir(directory)
            if os.path.isfile(os.path.join(directory, f)) and ("."+ext in f)]


def decompose_word(img, word, font):
    x_size = img.size[0]
    y_size = img.size[1]

    letters = len(word)
    array = img_to_array(img)
    array.shape = (array.shape[1], array.shape[2])
    #print word, img.size
    draw = Draw(Image.new("L",(100,100)))

    total_font_size = float(draw.textsize(word, font)[0])

    data = []
    start = 0
    for letter in word: #  range(0,x_size, x_size / len(word)):
        #print i*word_size,(i+1)*word_size, array.shape
        letter_size = draw.textsize(letter, font)[0]
        size = int(x_size*letter_size/total_font_size)
        middle = int((start + size) / 2)
        a = max(0,middle - 64)
        b = min(x_size, middle + 64)
        #print acc_size, acc_size+x
        img = array[:,a:b]

        if middle - 64 < 0:
            padding = 255*np.ones((y_size, 64 - middle))
            print padding.shape, img.shape
            img = np.hstack([padding,img])

        if middle + 64 > x_size:
            padding = 255*np.ones((y_size, (middle + 64) - x_size))
            print padding.shape, img.shape
            img = np.hstack([img,padding])

        data.append((img,letter))

        start = start + size

        #acc_size = acc_size + x

    return data

def load_data(path):
    files = list_files(path, 'png')
    print files
    #assert(0)
    xmls = list_files(path, 'xml')
    font = ImageFont.truetype('corsiva.ttf')
    data = []

    for xml in xmls:
        tree = ET.parse(xml)
        root = tree.getroot()
        for x in root.iter('word'):
            fid = x.attrib['id']
            ftext = x.attrib['text']
            if " " in ftext:
                continue

            for f in files:
                if fid+".png" in f:
                    print "Loading",f,"->",ftext
                    assert(not (" " in ftext))
                    data = data + decompose_word(load_img(f).convert('L'), ftext, font)
    return data

for i,(data,letter) in enumerate(load_data(sys.argv[1])):
  print letter, data.shape
  img = Image.fromarray(data[:, :].astype('uint8'), 'L') #array_to_img(data, "th")
  print letter, data.shape, img.size
  img.save("dump/img-"+letter+"-"+str(i)+".bmp")
  if i>50:
      break
  #pl.show()



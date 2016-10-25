 
import os
import re
import os.path
import argparse
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image, ImageFont, ImageOps, ImageEnhance
from PIL.ImageDraw import Draw
import base64

parser = argparse.ArgumentParser()
parser.add_argument('--source_data_path', type = str, required=True)
parser.add_argument('--dest_dir', type = str, required=True)

args = parser.parse_args()
print args
globals().update(vars(args))

font = ImageFont.truetype('corsiva.ttf')
#font = ImageFont.truetype('times.ttf')
draw = Draw(Image.new("L",(100,100)))

def load_list(fil, typ):
    return [(s.strip(),typ) for s in open(fil, 'r').readlines()]

def create_folders(path, conj):
    path = os.path.join(path,conj)
    if not os.path.exists(path):
        os.makedirs(path)
    for i in range(32,123):
        dir = os.path.join(path,str(i).zfill(3))
        if not os.path.exists(dir):
            os.makedirs(dir)

def find_chars_centers(word):
    text_width = float(draw.textsize(word['text'], font)[0])
    char_x = list()
    part=''
    for char in word['text']:
        part = part + char
        img_width = word['xmax']-word['xmin']
        char_width = draw.textsize(char, font)[0]
        part_width = draw.textsize(part, font)[0]
        text_pos = part_width - char_width/2
        char_pos = word['xmin']+int(round(float(img_width)*text_pos/text_width))
        char_x.append(char_pos)

    return char_x

def create_char_list(line,words):

    chars = list()
    line_text = line.attrib['text']
    line_text = line_text.replace('&quot;', '"')
    n = len(line_text) + 1
    line_text = ' ' + line_text + ' '
    i = 1
    nw = 0
    while i < n:
        element = line_text[i]
        if element == ' ':
            char = {'text': line_text[i-1:i+2],
                    'x_mid': int(0.5*(words[nw-1]['xmax']+words[nw]['xmin'])), # distancia media entre palabras
                    'value': ' ',
                    'word_id': words[nw]['id']}
            chars.append(char)
            i=i+1
        else:
            word_chars_x = find_chars_centers(words[nw])
            for j in range(len(word_chars_x)):
                char = {'text': line_text[i-1:i+2],
                        'x_mid': word_chars_x[j],
                        'value': line_text[i],
                        'word_id': words[nw]['id']}
                chars.append(char)
                i=i+1
            nw = nw + 1

    return chars
import time
def crop_and_save_line_chars(conj,form_image,chars,ymin,ymax,char_width,dest_dir):

    for char in chars:
        dir = os.path.join(dest_dir,conj,str(ord(char['value'])).zfill(3))
        filename = os.path.join(dir, base64.urlsafe_b64encode(char['text']) + '_' + char['word_id'] + '.png')
        xmin = char['x_mid'] - char_width/2
        xmax = xmin + char_width
        char_image = form_image.crop((xmin,ymin,xmax,ymax))
        resized_image = ImageOps.fit(char_image,(32,32), Image.ANTIALIAS)
        contrast = ImageEnhance.Contrast(resized_image)
        contrast.enhance(2).save(filename)
        #char_image.show()
        #time.sleep(2)

def list_files(directory, ext='jpg|jpeg|bmp|png'):
    return [os.path.join(directory, f) for f in os.listdir(directory)
            if os.path.isfile(os.path.join(directory, f)) and ("."+ext in f)]

#import matplotlib.pyplot as plt
def refine_line_bounds(form_image,xmin,ymin,xmax,ymax):
    line_im = form_image.crop((xmin,ymin,xmax,ymax))
    contrast = ImageEnhance.Contrast(line_im)
    line_np = 255 - np.array(contrast.enhance(2))
    histo = np.square(np.mean(line_np,axis=(1)))
    prob = histo / np.sum(histo)
    y = np.asarray(range(ymin,ymax))
    y_mean = np.dot(prob,y)
    s = (y - y_mean)**2
    s = np.sqrt(np.dot(prob,s))
    ymax = min(ymax,int(round(y_mean+4*s)))
    ymin = max(ymin,int(round(y_mean-4*s)))
    #print y.shape, histo.shape
    #plt.plot(y,histo)
    #plt.axvline(x=ymin)
    #plt.axvline(x=ymax)
    #plt.show()
    return ymin, ymax

xmls = sorted(list_files(source_data_path+'xml','xml'))

conjs=load_list('classification/train.txt', 'train')+load_list('classification/test.txt', 'test')+load_list('classification/valid.txt', 'valid')
conjs={re.sub(r"(.*)-.*", r'\1', x):y for x,y in conjs}
create_folders(dest_dir, 'train')
create_folders(dest_dir, 'test')
create_folders(dest_dir, 'valid')

it = 0
#print conjs
for xml in xmls:
    try:
      tree = ET.parse(xml)
      root = tree.getroot()
      form_id = root.attrib['id']
      form_path = source_data_path+'forms/'+form_id+'.png'
      form_image = Image.open(form_path)
      for line in root.iter('line'):
        line_text = line.attrib['text']
        line_id = line.attrib['id']
        ymin=100000
        ymax=0
        xmin=100000
        xmax=0
        words = list()
        for xword in line.iter('word'):
            wxmax=0
            wxmin=100000
            for comp in xword.iter('cmp'):
                y = int(comp.attrib['y'])
                h = int(comp.attrib['height'])
                x = int(comp.attrib['x'])
                w = int(comp.attrib['width'])
                ymin = min(ymin, y - 1)
                ymax = max(ymax, y + h)
                wxmin = min(wxmin,x)
                wxmax = max(wxmax,x+w)
            xmin = min(xmin, wxmin)
            xmax = max(xmax, wxmax)
            word = {'text': xword.attrib['text'],
                    'xmin': wxmin-1,
                    'xmax': wxmax,
                    'id': xword.attrib['id']}
            words.append(word)
        chars = create_char_list(line,words)
        ymin,ymax = refine_line_bounds(form_image,xmin,ymin,xmax,ymax)
        if form_id in conjs:
            crop_and_save_line_chars(conjs[form_id], form_image,chars,ymin,ymax,ymax-ymin,dest_dir)
        else:
            print 'La linea '+line_id+' no esta en ningun conjunto.'
    except:
      print "skip!"


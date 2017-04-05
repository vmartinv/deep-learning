# -*- coding: utf-8 -*-
import re
import os
import sys
import h5py
from utils import *
from paths import *

def list_files(directory):
    return [os.path.join(directory, f) for f in os.listdir(directory)
            if os.path.isfile(os.path.join(directory, f))]

is_allowed = lambda c: c.isalnum() or c in {' ', '.', ','}

def filter_line(line):
    if any([c in {u'ª', u'²', u'³', u'º', u'ß', u'ð', u'ø'} for c in line]):
        return True
    if "\t" in line or ";" in line or "<" in line  or ">" in line or len(line) < 32:
        return True
    if not(all([is_allowed(c) for c in strip_accents(line)])):
        return True

    res=not(all([in_dicc(w) for w in strip_accents(line).split(' ')]))
    return False

def load_data(path, max_lines = None):
    files = list_files(path)
    #~ print files
    data = []

    for f in files:
        print "processing",f
        with codecs.open(f, encoding='ISO-8859-1') as infile:
            i = 0
            for line in infile:
                if i == max_lines:
                    break
                line = re.sub(' +',' ',line)[:-1].strip()
                if not filter_line(line):
                    #x = x.decode("ISO-8859-1","decode")
                    data.append(strip_accents(line))
                    #alphabet.update(set)
                    i = i + 1
                    #del x

    return ' '.join(data)

text = load_data("src/spanish-rawdata", 7500)
print('corpus length:', len(text))

chars = sorted(list(set(text)))
print('total chars:', len(chars))

for letter in chars:
    if not letter.isalpha():
        p=text.find(letter)
        print letter,'=',text[p-20:p+20]
for l in chars:
    print l,


H5FOLDER = wikiCorpusPath()
if not os.path.exists(H5FOLDER):
    os.makedirs(H5FOLDER)
# Create a new file using defaut properties.
dataset = h5py.File(os.path.join(H5FOLDER, 'con_dict_7500lines.h5'),'w')

# Create a dataset under the Root group.
text = text.lower().encode("unicode_escape")

text = dataset.create_dataset("dataset", data=[text])

# Close the file before exiting
dataset.close()


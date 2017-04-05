from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility
from time import time, strftime, localtime
import os
import unicodedata
import codecs
import re


def strip_accents(s):
    if type(s) != unicode:
        s=s.decode('utf-8')
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                      if unicodedata.category(c) != 'Mn')


dicc={''}
for dicfile in ['diccionario-espanol.txt', 'diccionario-espanol2.txt']:
    with codecs.open(dicfile, encoding='utf-8') as myfile:
        for l in myfile.readlines():
            dicc.add(strip_accents(l.strip()).lower().encode('utf-8'))
dicc.add('')
def in_dicc(w):
    w=re.sub(r'[^(a-z)]', '', w.lower())
    res= w in dicc
    if w.endswith('s'):
        res|=w[:-1] in dicc
    return res

class NameGen(object):
    def __init__(self, base_name):
        self.name = base_name + '--' + strftime("%d-%b-%Y--%H-%M", localtime())

    def get_name(self):
        return self.name

    def get_file(self, dire, suffix):
        if not os.path.exists(dire):
            os.makedirs(dire)
        return os.path.join(dire, self.name + '--' + suffix)

    def get_model_file(self, suffix=''):
        return self.get_file("models", suffix)

    def get_history_file(self, suffix=''):
        return self.get_file("histories", suffix)

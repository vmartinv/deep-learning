from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility
from time import time, strftime, localtime
import os
import unicodedata

def strip_accents(s):    
    if type(s) != unicode:
        s=s.decode('utf-8')
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                      if unicodedata.category(c) != 'Mn')

class NameGen(object):
    def __init__(self, base_name):
        self.name = base_name + '--' + strftime("%d-%b-%Y--%H-%M", localtime())
        
    def get_name(self):
        return self.name
    
    def get_file(self, dire, suffix):
        if not os.path.exists(dire):
            os.makedirs(dire)
        return os.path.join(dire, self.name + '--' + suffix)
        
    def get_model_file(self, suffix):
        return self.get_file("models", suffix)
        
    def get_history_file(self, suffix):
        return self.get_file("histories", suffix)

import os
import numpy as np
from keras.models import load_model
from top_k_metric import top3
from keras import backend as K
import sys

import cPickle as pickle

modelfile = "models/lstm.h5"
model = load_model(modelfile)

# En este archivo se encuentran los dictionaries con la cual la LSTM fue entrenada
# Mapean un caracter a su posicion en los vectores one-hot y viceversa
fname='charIx.bin'
with open(fname,"rb") as fp:
    char_indices = pickle.load(fp)
    indices_char = pickle.load(fp)

maxlen = 100

ks = char_indices.keys()
seed = '                                                                                                   .'

# Dada una cadena de caracteres predichos, devuelve una funcion que toma como argumento un caracter y devuelve
# la probabilidad de ser el proximo. Si la longitud de la cadena 'anteriores' es menor a la requerida, se concatena con la semilla
def LSTM_Pred(anteriores):
    anteriores = filter(lambda x : x in ks, anteriores)
    if (len(anteriores) < maxlen):
        sentence = seed + anteriores
        sentence = sentence[-maxlen:]
    else:
        sentence = anteriores[-maxlen:]
    x = np.zeros((1, maxlen, len(char_indices)))
    for t, char in enumerate(sentence):
        x[0, t, char_indices[char]] = 1.
    preds = model.predict(x, verbose=0)[0]
    return (lambda x : preds[char_indices[x]] if (char_indices.has_key(x)) else 0)


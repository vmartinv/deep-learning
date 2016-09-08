import numpy as np
import matplotlib.pyplot as plt
from utils import *

#Leemos el archivo
sentences = parse_file('caperucita.txt')
cantcap = len(sentences)
sentences += parse_file('machinelearning.txt')

#Generamos la lista de palabras
words = [word.lower() for word in flatten(sentences)]

CANTIDAD_DE_PALABRAS = 1000

#Calculamos la frecuencia de cada palabra
words_with_freq = calc_freq(words)
most_frequents = list(map(lambda x:x[0], words_with_freq[:min(CANTIDAD_DE_PALABRAS, len(words_with_freq))]))

#Construimos el vector deseado
interesting_words = make_interesting_words_dict(most_frequents)
vec_sentences_a = np.array(list(map(lambda s: make_vector(s, interesting_words, lambda w,f:f), sentences)))

X = PCADimReduction(vec_sentences_a, 2)
plt.plot(*zip(*X[:cantcap]), marker='o', color='r', ls='') 
plt.plot(*zip(*X[cantcap:]), marker='o', color='b', ls='') 
plt.show()






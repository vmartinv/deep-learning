import numpy as np
import matplotlib.pyplot as plt
from utils import *

#Leemos el archivo
sentences = parse_file('nietzsche.txt')

#Generamos la lista de palabras
words = [word.lower() for word in flatten(sentences)]

print("Cantidad de oraciones: %d" % (len(sentences)))
print("Cantidad de palabras: %d" % (len(words)))
print("Cantidad de palabras distintas: %d" % (len(set(words))))

CANTIDAD_DE_PALABRAS = 1000

#Calculamos la frecuencia de cada palabra
words_with_freq = calc_freq(words)
CANTIDAD_DE_PALABRAS = min(CANTIDAD_DE_PALABRAS, len(words_with_freq))
most_frequents = list(map(lambda x:x[0], words_with_freq[:CANTIDAD_DE_PALABRAS]))
print("20 palabras mas frecuentes: %s" % (most_frequents[:min(CANTIDAD_DE_PALABRAS, 20)]))

#Construimos el vector deseado
interesting_words = make_interesting_words_dict(most_frequents)
vec_sentences = np.array(list(map(lambda s: make_vector(s, interesting_words, lambda w,f:f), sentences)))

#Reducimos la dimensionalidad y ploteamos
X = PCADimReduction(vec_sentences, 2)
plt.plot(*zip(*X), marker='o', color='r', ls='') 
plt.show()






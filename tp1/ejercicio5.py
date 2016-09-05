import re
from collections import defaultdict
import operator
import numpy as np
from math import log2

def calc_freq(l):
    frequencies = defaultdict(int)
    for e in l:
        frequencies[e.lower()] += 1
    return list(reversed(sorted(frequencies.items(),  key=lambda x: x[1])))

def flatten(l):
    return [e for sl in l for e in sl]

#Leemos el archivo
with open('nietzsche.txt', 'r') as myfile:
    text=myfile.read().replace('\n', ' ')

#Separamos las oraciones
sentences = text.replace('?', '.').replace('!', '.').split('.')
#Parseamos cada oracion, una oracion tiene que ser una lista de palabras
sentences = list(filter(None, map(lambda sentence: list(filter(None, re.findall('[a-zA-Z]+', sentence))), sentences)))

#Generamos la lista de palabras
words = [word.lower() for word in flatten(sentences)]

print("Cantidad de oraciones: %d" % (len(sentences)))
print("Cantidad de palabras: %d" % (len(words)))
print("Cantidad de palabras distintas: %d" % (len(set(words))))

CANTIDAD_DE_PALABRAS = 1000

    
#Apartado 4. a)

#Calculamos la frecuencia de cada palabra
frequencies = defaultdict(int)
for word in words:
    frequencies[word] += 1
words_with_freq = calc_freq(words)

CANTIDAD_DE_PALABRAS = min(CANTIDAD_DE_PALABRAS, len(words_with_freq))
most_frequents = list(map(lambda x:x[0], words_with_freq[:CANTIDAD_DE_PALABRAS]))

print("20 palabras mas frecuentes: %s" % (most_frequents[:min(CANTIDAD_DE_PALABRAS, 20)]))


#Construimos el vector deseado
interesting_words = {word:i for i,word in enumerate(most_frequents)}
def make_vector_a(sentence):
    l = np.array([0]*CANTIDAD_DE_PALABRAS)
    for word, freq_in_sentence in calc_freq(sentence):
        if word in interesting_words:
            l[interesting_words[word]] += freq_in_sentence
    return l

vec_sentences_a = np.array(list(map(make_vector_a, sentences)))
print(vec_sentences_a)


#Apartado 5. b)

cant_ap_per_sentence = dict(calc_freq(flatten(map(set, sentences))))

less_frequents = list(map(lambda x:x[0], words_with_freq[-CANTIDAD_DE_PALABRAS:]))
print("20 palabras menos frecuentes: %s" % (less_frequents[:min(CANTIDAD_DE_PALABRAS, 20)]))

#Construimos el vector deseado
interesting_words = {word:i for i,word in enumerate(less_frequents)}
def make_vector_b(sentence):
    l = np.array([0]*CANTIDAD_DE_PALABRAS)
    for word, freq_in_sentence in calc_freq(sentence):
        if word in interesting_words:
            l[interesting_words[word]] = freq_in_sentence * log2(len(sentences)/cant_ap_per_sentence[word])
    return l

vec_sentences_b = np.array(list(map(make_vector_b, sentences)))
print(vec_sentences_b)


#Ejercicio 6
def mat_covariance(v):
    v = np.array(list(map(lambda x: np.array(x - x.mean()),v)))
    m = v.shape[0]
    ret = v*0
    for i in range(m):
        ret += v[i][np.newaxis].transpose().dot(v[i][np.newaxis])
    return ret/m

def PCA(x):
    m = x.shape[0]
    cov = mat_covariance(x)
    eigva, eigenve = np.linalg.eig(cov)
    eig = sorted([(val, eigenve[:,i]) for i,val in enumerate(eigva)],  key=lambda x: -x[0])
    tolerancia = 1
    k=1
    s=eig[0][0]
    print(eig[0][0])
    sp = eigva.sum()
    print("sp: "+str(sp))
    while s/sp<tolerancia:
        print(k)
        print(s)
        print(eig[k][0])
        s+=eig[k][0]
        k+=1
    print(k)
    U = np.array(list(map(lambda x:x[1], eig)))
    x = U.transpose().dot(x.transpose())
    x = np.array(list(map(lambda x: x[0:k], x)))
    return x

test = np.array([ [1,0,0,] , [1,1,1] , [1,2,3] ])
print(mat_covariance(test))
print(PCA(test))
print(mat_covariance(PCA(test)))


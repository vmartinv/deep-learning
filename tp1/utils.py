import re
from collections import defaultdict
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

def calc_freq(l):
    frequencies = defaultdict(int)
    for e in l:
        frequencies[e.lower()] += 1
    return list(reversed(sorted(frequencies.items(),  key=lambda x: x[1])))

def flatten(l):
    return [e for sl in l for e in sl]

def sep_words(sentence):
    return list(filter(None, re.findall('[a-zA-Z]+', sentence)))

def parse_text(text):
    text = text.replace('\n', ' ')
    #Separamos las oraciones
    sentences = text.replace('?', '.').replace('!', '.').split('.')
    #Parseamos cada oracion, una oracion tiene que ser una lista de palabras
    sentences = list(filter(None, map(sep_words, sentences)))
    return sentences

def parse_file(filename):
    with open(filename, 'r') as myfile:
        text = myfile.read()
    return parse_text(text)

def make_interesting_words_dict(words):
    return {word:i for i,word in enumerate(words)}

def make_vector(sentence, interesting_words, weightfunc):
    l = np.array([0]*len(interesting_words))
    for word, freq_in_sentence in calc_freq(sentence):
        if word in interesting_words:
            l[interesting_words[word]] += weightfunc(word, freq_in_sentence)
    return l

def kNeighbours(v,X,k):
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='brute').fit(X)
    distances, indices = nbrs.kneighbors(np.array([v]))
    return indices[0]

def PCADimReduction(X,n):
    X = X-X.mean(axis=0)
    pca = PCA(n_components=n)
    pca.fit(X)
    return np.array( pca.transform(X) )

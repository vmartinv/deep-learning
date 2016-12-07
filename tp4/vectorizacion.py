'''Example script to generate text from Nietzsche's writings.

At least 20 epochs are required before the generated text
starts sounding coherent.

It is recommended to run this script on GPU, as recurrent
networks are quite computationally intensive.

If you try this script on new data, make sure your corpus
has at least ~100k characters. ~1M is better.
'''

from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import random
import sys
import h5py
import os

path = "dataseth5/lesslines.h5"
with h5py.File(path,'r') as hf:
    text = str(hf.get('dataset')[0]).decode("unicode_escape")
print('corpus length:', len(text))

chars = sorted(list(set(text)))
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# cut the text in semi-redundant sequences of maxlen characters
maxlen = 40
step = 3
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])
print('nb sequences:', len(sentences))

print('Vectorization...')
X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1


H5FOLDER = 'dataseth5'
if not os.path.exists(H5FOLDER):
    os.makedirs(H5FOLDER)
# Create a new file using defaut properties.
dataset = h5py.File(os.path.join(H5FOLDER, 'lesslines-vectorized.h5'),'w')

X = dataset.create_dataset("X", data=X)
y = dataset.create_dataset("Y", data=y)

# Close the file before exiting
dataset.close()

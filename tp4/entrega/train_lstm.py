from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys
import h5py
from utils import *
import os
from keras.models import load_model

loadM = False

print('Cargando dataset...')
path = "src/wikicorpus/con_dict_7500lines.h5"
with h5py.File(path,'r') as hf:
    text = str(hf.get('dataset')[0]).decode("unicode_escape")
print('corpus length:', len(text))

chars = sorted(list(set(text)))
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

print('Creando oraciones...')
# cut the text in semi-redundant sequences of maxlen characters
maxlen = 100
step = 31
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])
print('nb sequences:', len(sentences))

print('Vectorizando...')
X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1


# build the model: a single LSTM
print('Creando modelo...')
model = Sequential()
model.add(LSTM(128, name='lstm1-128', consume_less='gpu', input_shape=(maxlen, len(chars)), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(128, name='lstm2-128', consume_less='gpu', return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(256, name='lstm3-256', consume_less='gpu', return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(256, name='lstm4-256', consume_less='gpu'))
model.add(Dropout(0.2))

model.add(Dense(256, name='densa_extra'))
model.add(Dropout(0.3))
model.add(Dense(len(chars), name='softmax', activation='softmax'))

if loadM:
    sys.stdout.write('Cargando pesos desde archivo...')
    sys.stdout.flush()
    model.load_weights('models/red_alvi_labdcc_moreNeurons.py--23-Dec-2016--18-57--iter58loss[1.2178814809178105]val_loss[1.1792419333715782].h5',by_name=False)
    print('OK')

optimizer = RMSprop(lr=0.01) #baje el lr de 0.01 a 0.0001
print('Compilando...')
model.compile(loss='categorical_crossentropy', optimizer=optimizer)


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

print('Entrenando...')
name=NameGen(os.path.basename(sys.argv[0]))
modelfile = name.get_model_file()
# train the model, output generated text after each iteration

best_val_loss = None
for iteration in range(1, 60):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    history = model.fit(X, y, batch_size=3072, nb_epoch=1, validation_split=0.25) #added validation
    if best_val_loss==None or history.history['val_loss']<best_val_loss:
        print('Guardando modelo en {}'.format(modelfile+'iter'+str(iteration)+'loss'+str(history.history['loss'])+'val_loss'+str(history.history['val_loss'])))
        best_val_loss = history.history['val_loss']
        model.save(modelfile+'iter'+str(iteration)+'loss'+str(history.history['loss'])+'val_loss'+str(history.history['val_loss'])+'.h5')

    start_index = random.randint(0, len(text) - maxlen - 1)

    for diversity in [0.4, 0.7, 0.9, 1.1]:
        print()
        print('----- diversity:', diversity)

        generated = ''
        sentence = text[start_index: start_index + maxlen]
        generated += sentence
        print('----- Generating with seed: "' + sentence + '"')
        sys.stdout.write(generated)
        for i in range(400):
            x = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(sentence):
                x[0, t, char_indices[char]] = 1.

            preds = model.predict(x, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]

            generated += next_char
            sentence = sentence[1:] + next_char

            sys.stdout.write(next_char)
            sys.stdout.flush()
        print()


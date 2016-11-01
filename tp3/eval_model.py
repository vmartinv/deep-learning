import base
from keras.models import load_model
from sys import argv
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import operator

THRESHOLD = 0.2

if len(argv)!=3:
    print("Usage python eval_model.py <model.h5> <db.h5>")
    exit(1)

model = load_model(argv[1])
X, Y = base.H5Dataset(argv[2]).get_data()
total = X.shape[0]
cases = defaultdict(lambda: ([], []))

acc={}
cant={}

def get_acc(name, X, Y, verbose=0):
    print("Evaluando {}...".format(name))
    acc = model.evaluate(X, Y, verbose=verbose)[1]
    print "\r"+" "*80+"\r",
    print("Cantidad: {}".format(X.shape[0]))
    print("Accuracy: {:.2f}".format(acc))
    print
    return acc

def save_graph():
    plt.rcParams["figure.figsize"] = [8, 12]
    y_pos = np.arange(len(acc))
    asdf = [(ch, ac, q) for (ch,ac),(_,q) in zip(acc.iteritems(), cant.iteritems())]
    asdf = sorted(asdf, key=operator.itemgetter(2))
    
    plt.barh(y_pos, [ac for __, ac, _ in asdf], align='center', alpha=0.4)
    plt.yticks(y_pos, [ch for ch, __, _ in asdf])
    plt.xlabel('Accuracy')
    plt.ylabel('Caracter')
    plt.grid(True)
    plt.title('')
    graphfile=argv[1].replace('--model.h5', '--accbychar.png')
    for i, (ch, a, q) in enumerate(asdf):
        plt.text(a + 0.01, i, "%.2f%%"%(q/float(total)*100.), color='black', fontweight='bold')
    plt.savefig(graphfile, bbox_inches='tight', dpi = 300)
    plt.clf()
    
acc['Todos'] = get_acc('todo el dataset', X, Y, verbose=1)
cant['Todos'] = total

print('Separando dataset...')
for x,y in zip(X, Y):
    cases[base.to_categorical(y)][0].append(x)
    cases[base.to_categorical(y)][1].append(y)
cases = {ch:(np.array(X), np.array(Y)) for ch,(X,Y) in cases.iteritems() if len(X)/float(total)*1000>=THRESHOLD}

cases = sorted(cases.items(), key=lambda (__, (_, Y)):Y.shape[0], reverse=True)
for i, (nch, (X, Y)) in enumerate(cases):
    ch = chr(nch+32)
    print "({:.0f}%)".format(i/float(len(cases))*100.),
    cacc = get_acc('caracter {}'.format(ch), X, Y)
    acc[ch]=cacc
    cant[ch]=X.shape[0]
        
    if ' ' in acc:
        acc['Espacios'] = acc[' ']
        cant['Espacios'] = cant[' ']
        del acc[' ']
        del cant[' ']
    
save_graph()

print(acc)


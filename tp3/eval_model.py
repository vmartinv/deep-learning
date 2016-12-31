import base
from keras.models import load_model
from sys import argv
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import operator
from keras.models import model_from_json
from top_k_metric import top3
from sklearn.metrics import confusion_matrix
import os

THRESHOLD = 0.2
USAGE = "Usage python eval_model.py <db.h5> (<model.h5> | <model.json> <weights.h5>)"
if len(argv)==3:
    model = load_model(argv[2])
elif len(argv)==4:
    with open(argv[2], "r") as text_file:
        json = text_file.read()
        model = model_from_json(json)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy', top3 ])
    model.summary()
    model.load_weights(argv[3])
else:
    print(USAGE)
    exit(1)

X, Y = base.H5Dataset(argv[1]).get_XY()
total = X.shape[0]
cases = defaultdict(lambda: ([], []))

acc={}
cant={}

def get_acc(name, X, Y, verbose=0):
    print("Evaluando {}...".format(name))
    score = model.evaluate(X, Y, verbose=verbose)
    acc = score[1]
    print "\r"+" "*80+"\r",
    print("Cantidad: {}".format(X.shape[0]))
    print("Accuracy: {:.4f}".format(acc))
    if len(score)==3:
        top3 = score[2]
        print("Top3: {:.4f}".format(top3))
    print
    return acc
    
acc['Todos'] = (get_acc('todo el dataset', X, Y, verbose=1), total)

print('Separando dataset...')
for x,y in zip(X, Y):
    cases[base.to_categorical(y)][0].append(x)
    cases[base.to_categorical(y)][1].append(y)
cases = {ch:(np.array(X), np.array(Y)) for ch,(X,Y) in cases.iteritems() if len(X)/float(total)*1000>=THRESHOLD}

cases = sorted(cases.items(), key=lambda (__, (_, Y)):Y.shape[0], reverse=True)
proccessed = 0
for i, (nch, (Xp, Yp)) in enumerate(cases):
    ch = chr(nch+32)
    print "({:.0f}%)".format(proccessed/float(total)*100.),
    cacc = get_acc('caracter {}'.format(ch), Xp, Yp)
    acc[ch]=(cacc, Xp.shape[0])
    proccessed += Xp.shape[0]
    if ' ' in acc:
        acc['Espacios'] = acc[' ']
        del acc[' ']
print('(100%)')
print(acc)

graphfile=argv[2].replace('--model-train-weights', '').replace('--model', '')
graphfile=graphfile.replace('models/', 'histories/')
graphfile=graphfile.replace(os.path.splitext(graphfile)[1], '--accbychar.png')
print('Creando grafico {}...'.format(graphfile))
plt.rcParams["figure.figsize"] = [8, 12]
y_pos = np.arange(len(acc))
asdf = [(ch, ac, q) for ch,(ac,q) in acc.iteritems()]
asdf = sorted(asdf, key=operator.itemgetter(2))

plt.barh(y_pos, [ac for __, ac, _ in asdf], align='center', alpha=0.4)
plt.yticks(y_pos, [ch for ch, __, _ in asdf])
plt.xlabel('Accuracy')
plt.ylabel('Caracter')
plt.grid(True)
plt.title('')
for i, (ch, a, q) in enumerate(asdf):
    plt.text(a + 0.01, i, "%.2f%%"%(q/float(total)*100.), color='black', fontweight='bold')
plt.saveevfig(graphfile, bbox_inches='tight', dpi = 300)
plt.clf()


def myConfusion(X,Y,Ypredict):
    Yclases = []
    for yi in Y:
        Yclases.append(np.argmax(yi))

    YPredictclases = []
    for yi in Ypredict:
        YPredictclases.append(np.argmax(yi))

    return confusion_matrix(Yclases,YPredictclases)

Ypredict = model.predict(X)
CM = myConfusion(X,Y,Ypredict)
tolerancia = 0.1
for i in range(len(CM)):
	actual = []
	for j in range(len(CM[i])):
		if i != j and CM[i][j]> tolerancia*sum(CM[i]):
			print( "%s %s %.2f" %(chr(i+32), chr(j+32), CM[i][j]/float(sum(CM[i])) ))




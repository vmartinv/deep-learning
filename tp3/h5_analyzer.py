import base
from sys import argv
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import operator
from collections import defaultdict

THRESHOLD = 0.4
GRAPHFILE='per-char-analysis.png'

if len(argv)!=2:
    print("Usage python h5_analyzer.py <db.h5>")
    exit(1)

X, Y = base.H5Dataset(argv[1], "Conjunto").get_XY()
total = X.shape[0]

cant = defaultdict(int)
for y in Y:
    cant[chr(base.to_categorical(y)+32)] += 1
    
cant['Espacios']=cant[' ']
del cant[' ']

cant = sorted(cant.items(), key=operator.itemgetter(1))
for i,(c,q) in enumerate(cant):
    print("(#{}) {}: {} ({:.2f}%)".format(len(cant)-i, c, q, q/float(total)*100.))
print('Total: {}'.format(total))

print(list(reversed(cant)))

print('Guardando grafico en {}...'.format(GRAPHFILE))
qotros = sum([q for _,q in cant if q/float(total)*100<THRESHOLD])
cant = [(c,q) for c,q in cant if q/float(total)*100>=THRESHOLD]
cant = sorted(cant+[('Otros', qotros)], key=operator.itemgetter(1))

cant = list(reversed(cant))
cant = [(c,q/float(total)*100.) for c,q in cant]
C, Q = zip(*cant)

#~ print(plt.rcParams["figure.figsize"])
#~ plt.rcParams["figure.figsize"] = [8, 15]

y_pos = np.arange(len(cant))
plt.barh(y_pos, Q, align='center', alpha=0.4)
plt.yticks(y_pos, C)
plt.xlabel('Porcentaje del dataset que representa')
plt.ylabel('Caracter')
plt.grid(True)
plt.title('')
for i, q in enumerate(Q):
    plt.text(q + 0.01, i, "{:.2f}%".format(q), color='black', fontweight='bold')
plt.savefig(GRAPHFILE, bbox_inches='tight', dpi = 300, figsize=(100, 100))
plt.clf()


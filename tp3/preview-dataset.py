from __future__ import print_function
import numpy as np
from sys import argv
np.random.seed(1337)  # for reproducibility
import matplotlib as mpl
import base
mpl.use('Agg')
import matplotlib.pyplot as plt
from pylab import *

USAGE = "Usage python preview_datset.py <db.h5>"
if len(argv)!=2:
    print(USAGE)
    exit(1)

X, _ = base.H5Dataset(argv[1]).get_XY()

def prepare_show(face):
    m = face.min()
    M = face.max()
    return (face-m)/(M-m)

graphfile="preview2.png"
print ("Guardando vista previa en", graphfile, "...")
COLS=5
ROWS=5
ax = plt.gca()
ax.patch.set_facecolor('black')
for i in range(ROWS):
    for j in range(COLS):
        sub1 = plt.subplot(ROWS, COLS, 1+i*COLS+j)
        fig = plt.imshow(prepare_show(X[i*COLS+j].reshape(32, 32)),
                      cmap=plt.get_cmap('gray'))
        plt.axis('off')
        autoAxis = sub1.axis()
        rec = Rectangle((autoAxis[0]-0.7,autoAxis[2]-0.2),(autoAxis[1]-autoAxis[0])+1,(autoAxis[3]-autoAxis[2])+0.4,fill=False,lw=1)
        rec = sub1.add_patch(rec)
        rec.set_clip_on(False)
        
# muestra el plot
plt.savefig(graphfile, bbox_inches='tight', dpi = 300)
plt.clf()
    


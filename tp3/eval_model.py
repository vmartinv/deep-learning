import base
from keras.models import load_model
from sys import argv
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

if len(argv)!=2:
    print("Usage python eval_model.py <model.h5>")
    exit(1)

model = load_model(argv[1])

acc={}

data = base.H5Dataset("dataseth5/train.h5", "Todo el conjunto de training")
acc['Todos'] = data.evaluate(model)[1]
total = data.get_data()[0].shape[0]

for ch in range(0, 91):
    data = base.H5Dataset("dataseth5/train.h5", "Caracter " + chr(ch+32))
    data.filter(lambda _, y: base.to_categorical(y)==ch)
    name = "preview"+str(ch+32).zfill(3)
    #~ print("Guardando vista previa en %s..."%(name))
    #~ data.preview(name)
    score = data.evaluate(model)
    if score is not None and score[1]*data.get_data()[0].shape[0]>1000:
        acc[chr(ch+32)]=score[1]*data.get_data()[0].shape[0]/total
        
    if ' ' in acc:
        acc['espacio'] = acc[' ']
        del acc[' ']

    y_pos = np.arange(len(acc))
    plt.barh(y_pos, acc.values(), align='center', alpha=0.4)
    plt.yticks(y_pos, acc.keys())
    plt.xlabel('Accuracy/Cantidad')
    plt.ylabel('Caracter')
    plt.grid(True)
    plt.title('Accuracy por caracter')
    graphfile=argv[1].replace('--model.h5', '--accbychar.png') 
    plt.savefig(graphfile, bbox_inches='tight', dpi = 300)
    plt.clf()
    
print(acc)


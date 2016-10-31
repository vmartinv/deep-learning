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
cant={}

data = base.H5Dataset("dataseth5/train.h5", "Todo el conjunto de training")
total = data.get_data()[0].shape[0]
acc['Todos'] = data.evaluate(model)[1]
cant['Todos'] = total

for ch in range(0, 91):
    data = base.H5Dataset("dataseth5/train.h5", "Caracter " + chr(ch+32))
    data.filter(lambda _, y: base.to_categorical(y)==ch)
    name = "preview"+str(ch+32).zfill(3)
    #~ print("Guardando vista previa en %s..."%(name))
    #~ data.preview(name)
    score = data.evaluate(model)
    if score is not None and data.get_data()[0].shape[0]>1000:
        acc[chr(ch+32)]=score[1]
        cant[chr(ch+32)]=data.get_data()[0].shape[0]
        
    if ' ' in acc:
        acc['Espacios'] = acc[' ']
        cant['Espacios'] = cant[' ']
        del acc[' ']
        del cant[' ']

    y_pos = np.arange(len(acc))
    plt.barh(y_pos, acc.values(), align='center', alpha=0.4)
    plt.yticks(y_pos, acc.keys())
    plt.xlabel('Accuracy')
    plt.ylabel('Caracter')
    plt.grid(True)
    plt.title('Accuracy por caracter')
    graphfile=argv[1].replace('--model.h5', '--accbychar.png')
    for i, (a, q) in enumerate(zip(acc.values(), cant.values())):
        plt.text(a + 0.01, i, "%.2f%%"%(q/float(total)*100.), color='black', fontweight='bold')
    plt.savefig(graphfile, bbox_inches='tight', dpi = 300)
    plt.clf()

    
    
    
    plt.clf()
    
print(acc)


import os
import numpy as np
import preproceso as pp
from keras.models import model_from_json
from top_k_metric import top3
from keras import backend as K
import paths
import genventana as genventana
import costprediction

def applyCNN(fileName):
    img = pp.preprocessOne(fileName)
    name, wins = genventana.windows(img,32,2,32)

    json = open('models/tp3Net-model.json').read()
    tp3Net = model_from_json(json)
    tp3Net.load_weights('models/tp3Net-weights.h5')
    tp3Net.compile(loss='categorical_crossentropy',
                   optimizer='adadelta',
                   metrics=['accuracy', top3 ])

    classes = 100*tp3Net.predict_proba(wins, batch_size=32)
    cantwindows , _ = classes.shape

    if paths.previewPath():
        prediccionOut = ''.join( map(chr, 32+tp3Net.predict_classes(wins, batch_size=32))) + '\n'
        nroLinea = os.path.basename(fileName)[0:11]
        original = os.path.basename(fileName)[12:-4]
        prediccionesCNN = open(  os.path.join(paths.previewPath() , 'predicciones.txt'), 'a')
        prediccionesCNN.write( "NROLINEA: " + nroLinea + "\n")
        prediccionesCNN.write( "ORIGINAL: " + original + "\n")
        prediccionesCNN.write( 'PREDICCION : ' + prediccionOut + "\n" )
        prediccionesCNN.write( "COSTO 1: {}\n".format(costprediction.costFunc(original,prediccionOut)*1000.0 / cantwindows ))
        prediccionesCNN.write( "COSTO 2: {}\n".format(costprediction.costFunc2(original,prediccionOut)*1000.0 / cantwindows ))
        prediccionesCNN.write( "COSTO 3: {}\n".format(costprediction.costFunc3(original,classes)*1000.0 / cantwindows ))

        prediccionesCNN.write( ' --------------------------------------------------------\n\n')
        prediccionesCNN.close()
    return classes




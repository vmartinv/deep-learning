import os
import numpy as np
import preproceso as pp
from keras.models import model_from_json
from top_k_metric import top3
from keras import backend as K

source = "../rawdata/lines/"
dest = "preview2/"
img_rows = 32
img_cols = 32


def windows(nameImg,winWidth,step,finalWidth):
    name,img=nameImg
    wins=[]
    w,h = img.size
    for s in range(0,w-winWidth,step):
        win = img.crop((s,0,s+winWidth,h))
        if winWidth < finalWidth:
            diffL=(finalWidth-winWidth)//2
            diffR=(finalWidth-winWidth+1)//2
            win=pp.fill_border(win,diffL,diffR,img_rows)
        if not os.path.exists(os.path.join(dest,'wins', name[0:-4])):
            os.makedirs(os.path.join(dest,'wins', name[0:-4]))
        win.save(os.path.join(dest,'wins', name[0:-4], 'step'+str(s)+'.png'))
        win = np.asarray(win,dtype=np.float32)
        win /= 255
        win -= 0.87370783 #featurewise mean de los datos de tp3
        wins.append(win)
    return (name,np.array(wins))

def reshape(imgs):
    if K.image_dim_ordering() == 'th':
        imgs = imgs.reshape(imgs.shape[0], 1, img_rows, img_cols)
    else: #'tf'
        imgs = imgs.reshape(imgs.shape[0], img_rows, img_cols, 1)
    return imgs

imgs=pp.preprocess(source,dest)
wins=map(lambda i: windows(i,32,2,32),imgs)
wins=map(lambda (n,i): (n,reshape(i)), wins)
winsImgsOnly = np.array([i for n,i in wins])

json = open('tp3Net/tp3Net-model.json').read()
tp3Net = model_from_json(json)
tp3Net.load_weights('tp3Net/tp3Net-weights.h5')
tp3Net.compile(loss='categorical_crossentropy',
               optimizer='adadelta',
               metrics=['accuracy', top3 ])

#~ tp3Net.summary()

classes = map(lambda (n,i): (n, 32+tp3Net.predict_classes(i, batch_size=32)),wins)
res= map(lambda (n, s): n[:-4]+'='+''.join(map(chr,s)),classes)
print('\n\n'.join(res))

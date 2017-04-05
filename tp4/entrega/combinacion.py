import os
import numpy as np
import sys
import lstm




def combinador1(clases):
    cw, cc = clases.shape
    acum = np.empty_like (clases[0])

    out = ""
    LSTM = lstm.LSTM_Pred('')

    for i in range(0,cw):
        maxj = 0
        for j in range(0,cc):
            if clases[i][maxj]<clases[i][j]:
                maxj = j

        hayQueEscribir = ( out=="" and (clases[i][maxj]>=0.5 or i>=4) ) or ( out!="" and (clases[i][ord(out[-1])-32]<0.5*clases[i][maxj] or LSTM(out[-1].lower())>=0.5) )
        if hayQueEscribir:
            maxacum = 0
            acum[0] += clases[i][0]*1.15 # alto parche
            for j in range(0,cc):
                acum[j] += clases[i][j]
                maxacum = max(maxacum,acum[j])

            pred = '$'
            maxprob = 0
            for j in range(0,cc):   
                myprob = acum[j]*0.8 + 0.2*LSTM(chr(j+32).lower())*100
                if acum[j]>=0.2*maxacum and  myprob>maxprob:
                    maxprob = myprob
                    pred = chr(j+32)


            if maxprob>=0.4*cw:
                out += pred
                LSTM = lstm.LSTM_Pred(out)
                for j in range(0,cc):
                    acum[j] = 0
    return out




def combinador2(clases):
    cw, cc = clases.shape
    acum = np.empty_like (clases[0])

    out = ""
    LSTM = lstm.LSTM_Pred('')

    lastspace = -1
    for i in range(0,cw):
        maxj = 0
        for j in range(0,cc):
            if clases[i][maxj]<clases[i][j]:
                maxj = j

        hayQueEscribir = ( out=="" and (clases[i][maxj]>=0.5 or i>=4) ) or ( out!="" and (clases[i][ord(out[-1])-32]<0.5*clases[i][maxj] or LSTM(out[-1].lower())>=0.5) )
        if hayQueEscribir:
            totacum = maxacum = 0
            acum[0] += clases[i][0]*1.15 # alto parche
            for j in range(0,cc):
                acum[j] += clases[i][j]
                maxacum = max(maxacum,acum[j])
                totacum += acum[j]

            pred = '$'
            maxprob = 0
            for j in range(0,cc):   
                lstmprob = LSTM(chr(j+32).lower())
                myprob = acum[j]/totacum * 0.7 + lstmprob * 0.3
                if ((acum[j]>=0.5*cw or  myprob>=0.8) and myprob>maxprob) or (False if len(out)-lastspace<4 else lstmprob>0.5):
                    maxprob = myprob
                    pred = chr(j+32)


            if pred!='$':
                out += pred
                if not pred.isalpha(): lastspace = len(out)
                LSTM = lstm.LSTM_Pred(out)
                for j in range(0,cc):
                    acum[j] = 0
    return out



def simpleCombinador(P1,P2,P3):
    def simple(clases):
        cw, cc = clases.shape
        acum = np.empty_like (clases[0])
        out = ""
        last = -1
        LSTM = lstm.LSTM_Pred(out)
        for i in range(0,cw):
            acum += clases[i]
            acum[0] += clases[i][0]*P1 
            if last>0:
                acum[last] += - P2*clases[i][last] + 100*(1-P2)*LSTM(chr(last+32))
            lstm_probs = np.array(map( lambda x : LSTM(chr(x+32))*100 , range(0,cc) )) 
            combined_probs = acum*0.8 + lstm_probs*0.2
            j = np.argmax(combined_probs)
            if combined_probs[j]>=P3*cw:
                out += chr(j+32)
                last = j
                LSTM = lstm.LSTM_Pred(out)
                acum = acum*0
        return out
    return simple
        

nuevoscombinadores = []
for P2 in [0.4, 0.5, 0.6, 0.7]:
    for P3 in [0.5, 0.6, 0.65, 0.7]:
        nuevoscombinadores.append( ("nuevocombinador{}-{}".format(P2,P3),simpleCombinador(0.2,P2,P3)))

combinadores = [('combinador1', combinador1), ('combinador2', combinador2)] #+ nuevoscombinadores



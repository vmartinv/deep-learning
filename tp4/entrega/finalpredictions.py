import paths
import combinacion
import os
import numpy as np
import unidecode


def editDistance(s1, s2):
    s1 = (unidecode.unidecode(s1.decode('utf-8')) ).lower()
    s2 = (unidecode.unidecode(s2.decode('utf-8')) ).lower()
    len1, len2 = len(s1), len(s2)
    dp = np.zeros((len1+1, len2+1), int)
    for i in range(len1,-1,-1):
        for j in range(len2,-1,-1):
            if i==len1 and j==len2:
                dp[i][j]=0
            elif i==len1:
                dp[i][j] = len2-j
            elif j==len2:
                dp[i][j] = len1-i
            elif s1[i]==s2[j]:
                dp[i][j] = dp[i+1][j+1]
            else:
                dp[i][j] = min(dp[i+1][j], dp[i+1][j+1], dp[i][j+1]) + 1
    return dp[0][0]


list_npy = [os.path.join(paths.npyPath(), f) for f in os.listdir(paths.npyPath())]
costAcc = np.zeros(len(combinacion.combinadores))
totalLen = 0
for f in list_npy:
    clases = np.load(f)
    resultfile = open(os.path.join( paths.predictionPath(), os.path.basename(f)[0:-4]+'.txt'), "w")
    original = os.path.basename(f)[12:-4]
    totalLen += len(os.path.basename(f)[12:-4])
    # print("DEBUG: ", original)
    resultfile.write( "ORIGINAL: " +original + "\n")
    idcombinador=0
    for namecomb, fcomb in combinacion.combinadores:
        out = fcomb(clases)
        resultfile.write( namecomb + ": " + out + "\n")
        costoED = editDistance( original , out)
        costAcc[idcombinador] += costoED
        resultfile.write( "EDIT-DISTANCE: " + "{}".format(costoED) + "\n")
        idcombinador+=1
    resultfile.close()


for idcombinador in range(0,len(combinacion.combinadores)):
    print( combinacion.combinadores[idcombinador][0], ": ", costAcc[idcombinador] , " // ",  costAcc[idcombinador] / totalLen )



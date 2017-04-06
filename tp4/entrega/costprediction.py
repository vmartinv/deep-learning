import numpy as np
import unidecode

def costChar(c1, c2):
    return (1 if c1 != c2  else 0)


def isAMatch(c1, c2):
    return (1 if c1 == c2  else 0)

# Esta funcion de costo es similar a una distancia edicion (minimo numero de borrados, ediciones o inserciones) de la cadena original (s1) a la predicha.
# Decimos similar puesto a que permitimos a varios caracteres de s2 matchearse con uno de s1 (por los caracteres repetidos sobre la prediccion).
def costFunc(s1, s2):
    s1 = (unidecode.unidecode(s1.decode('utf-8')) ).lower()
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
            else:
                dp[i][j] = min(dp[i+1][j], dp[i+1][j+1], dp[i][j+1]) + costChar( s1[i], s2[j] )  
    return dp[0][0]


# Es una especie de distancia edicion pero penaliza mas no predecir un caracter que esta en la cadena original.
# Resuelve el problema de que: aaaaaabbbbbbbbccccccccc sea una buena prediccion de axbxc (las x no son predichas cuando si debieran serlo! y eso es grave!)
# Lo que queremos decir que no es el mismo costo las ediciones sobre s1 que sobre s2.

def costFunc2(s1, s2):
    s1 = (unidecode.unidecode(s1.decode('utf-8')) ).lower()
    len1, len2 = len(s1), len(s2)
    dp = np.zeros((len1+1, len2+1, 2), int)
    costoSalto = 20
    for i in range(len1,-1,-1):
        for j in range(len2,-1,-1):
            for st in range(1,-1,-1):
                if i==len1 and j==len2:
                    dp[i][j][st]=0
                elif i==len1:
                    dp[i][j][st] = len2-j
                else:
                    dp[i][j][st] = dp[i+1][j][0] + (1 if j==len2 else costChar(s1[i],s2[j])) *  (1 if st==1 else costoSalto)
                    if j<len2:
                        dp[i][j][st] = min(dp[i][j][st], dp[i+1][j+1][0] + costChar(s1[i],s2[j]) *  (1 if st==1 else costoSalto) ) 
                        dp[i][j][st] = min(dp[i][j][st], dp[i][j+1][1 if st==1 else isAMatch(s1[i],s2[j])] + costChar(s1[i],s2[j]) *  (1 if st==1 else costoSalto)  )
    return float(dp[0][0][0])/costoSalto

# Esta es en nuestro criterio la mejor funcion de costo ya que no tiene en cuenta la prediccion de caracter mas probable sino toda la matriz de probabilidades que es output de la CNN.
# Trata de matchear o justificar la distribucion de probabilidad de cada ventana asignandole un caracter de la cadena original. El costo de esta justificacion esta dado por la probabilidad de que esta justificacion (o matcheo) sea incorrecto segun esa distribucion. Ademas penaliza igual que las anteriores el hecho de que haya caracteres en la original que no sean justificados (predichos) por ninguna ventana.
# A nuestro criterio es la que tiene mas correlacion con los resultados finales. En el sentido que cuando es baja esta funcion de costo, los resultados finales son positivos.

def costFunc3(original, probpredict):
    original = (unidecode.unidecode(original.decode('utf-8')) ).lower()

    len2, _ = probpredict.shape
    len1 = len(original)
    costoSalto = 1
    dp = np.zeros((len1+1, len2+1, 2), np.float)
    for i in range(len1,-1,-1):
        for j in range(len2,-1,-1):
            for st in range(1,-1,-1):
                if i==len1 and j==len2:
                    res = 0
                else:
                    res = 99999999
                if i<len1:
                    res = min(res, dp[i+1][j][0] + (0 if st==1 else costoSalto) )
                if j<len2:
                    if i<len1 and ord(original[i])-32>=0 and ord(original[i])-32<91:
                        probMatch = max(probpredict[j][ord(original[i])-32]/100, probpredict[j][ord(original[i].upper())-32]/100)
                        res = min(res, (1-probMatch) + dp[i][j+1][ 1 if st==1 or probMatch>0.5 else 0 ])
                    else:
                        res = min(res,  1+dp[i][j+1][0])
                dp[i][j][st] = res
    return dp[0][0][0]




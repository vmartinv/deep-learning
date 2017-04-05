import os

def auxcheck(path):
    if path and not os.path.exists(path):
        os.makedirs(path)

def previewPath():
    auxcheck(__previewPath)
    return __previewPath
def npyPath():
    auxcheck(__npyPath)
    return __npyPath
def srcPath():
    auxcheck(__srcPath)
    return __srcPath
def predictionPath():
    auxcheck(__predictionPath)
    return __predictionPath
def rawDataPath():
    auxcheck(__rawDataPath)
    return __rawDataPath
def wikiCorpusPath():
    auxcheck(__wikiCorpusPath)
    return __wikiCorpusPath

__previewPath = "./preview"
__npyPath = "./npyCNN"
__srcPath = "./src/newdataset"
__predictionPath = "./predictions"
__rawDataPath = "./src/spanish-rawdata"
__wikiCorpusPath = "./src/wikicorpus"

import pandas as pd 
from collections import Counter
from itertools import chain
import math

def main(fullDF, currentDF, granulation):
    
    resultTF = TF(currentDF["lemmas"])    
    preIDF = preprocessIDF(fullDF, currentDF, granulation)
    for word in resultTF:
        idf = IDF(preIDF, word)
        resultTF[word] = resultTF[word] * idf
    
    words = resultTF.most_common(10)
    
    result = ""
    for x in words:
        result = result + " '" + x[0] + "'"
    result = "TF-IDF words:" + result
        
    return result

def TF(series):
    allWords = chain.from_iterable(series)
    allWords = Counter(allWords)
    n = len(list(allWords))
        
    result = Counter({k:v/n for k,v in allWords.items()})
    
    return result

def preprocessIDF(big, small, granulation):
    
    result = big.drop(small.index)
    if granulation == 'Rating':
        result = result.drop(result.columns.difference(['stars','lemmas']), 1)
        result = result.groupby('stars')['lemmas'].sum()
    else:
        result = result.drop(result.columns.difference(['date', 'lemmas', 'stars']), 1)    
        if granulation == 'Year':
            result = result.drop(result.columns.difference(['date', 'lemmas']), 1)
            result["date"] = pd.to_datetime(result["date"]).dt.year
        elif granulation == 'Month':
            result = result.drop(result.columns.difference(['date', 'lemmas']), 1)
            result["date"] = pd.to_datetime(result["date"])
            result["date"] = result["date"].map(lambda x: x.strftime('%Y-%m'))   
        elif granulation == 'YearPlus':
            result["date"] = pd.to_datetime(result["date"]).dt.year.map(str) + "#" + result["stars"].map(str)
        elif granulation == 'MonthPlus':
            result["date"] = pd.to_datetime(result["date"])
            result["date"] = result["date"].map(lambda x: x.strftime('%Y-%m')).map(str) + "#" + result["stars"].map(str)
        result = result.groupby('date')['lemmas'].sum()
    
    result = result.apply(set)
    
    return result

def IDF(item, word):
    
    N = len(item) + 1
    DFt = 1
    for x in item:
        if word in x:
            DFt = DFt + 1
    result = math.log(N/DFt)
    
    return result
import pandas as pd
import nltk
import re
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer

stopCounter = {}

def main(item, common, top, granulation, getLemma, frequency, period):
    
    if granulation == "Restaurant" or granulation == "restaurant":
        period = ""
    mid = prepareData(item, granulation, getLemma)
    result = frameY(mid, top, common, frequency, period)
    
    return result

def readCSV(item):
    data = pd.read_csv(item, dtype={0: int}, index_col=0, encoding='latin-1')
    df = pd.DataFrame(data)
    return df

def filterTime(item,time):
    
    result = item.copy()
    indexNames = item[item['time'] != time].index
    result.drop(indexNames , inplace=True)
    result = result.rename(columns={'text': str(time)})
    
    return result[str(time)]

def filterPast(item, time):
    
    result = item.copy()
    indexNames = item[pd.to_datetime(item['time']) > pd.to_datetime(time)].index
    result.drop(indexNames , inplace=True)
    result = result.rename(columns={'text': str(time)})
    
    return result[str(time)]

def prepareData(item, granulation, wordsType):
    
    if (granulation == "Restaurant" or granulation == "restaurant"):
        item["newID"]  = item["name"].map(str) +" - "+ item["postal_code"].map(str)
        item = item.loc[:, ['newID']]
    else:
        item = item.loc[:, ['date']]    
        item = item.sort_values('date', ascending = True)
        
    if granulation == "Year":
        item['time'] = pd.to_datetime(item.date).dt.year
    elif (granulation == "Month"):
        item['time'] = pd.to_datetime(item.date)
        item['time'] = item.time.map(lambda x: x.strftime('%Y-%m'))
        
    elif (granulation == "Restaurant"):
        item['time'] = item.newID# + item.name.postal_code
        
    result = item.loc[:, ['time']]    
    
    wordsDF = readCSV('static/revSamTok.csv')   
    
    result = pd.merge(result, wordsDF[wordsType], left_index=True, right_index=True, how='left')
    result.rename(columns={wordsType: 'text'}, inplace=True)
    
    return result

def frameY(item, top, common, frequency, period):
    start = item.rename(columns={'text': "Overall"})
    
    if frequency == "True" or frequency == "true":
        start = td_idf(start["Overall"], top)
    else: 
        start = tokenCount(start["Overall"], top, common)
    
    uniqueTime = item.time.unique() 
    
    if period == "True" or period == "true":
        DF_list = [filterPast(item, i) for i in uniqueTime]
    else:
        DF_list = [filterTime(item, i) for i in uniqueTime]
    
    if frequency == "True" or frequency == "true":
        mid = pd.concat([td_idf(i, top) for i in DF_list], axis=1)
    else:    
        mid = pd.concat([tokenCount(i, top, common) for i in DF_list], axis=1)
    
    result = pd.concat([start,mid], axis=1)
    
    result.index+=1
    result.index.names = ['Place']

    return result

def tokenCount(item, top, common):
    global stopCounter
    
    allWords = ''.join(item)
    allWords = allWords.replace("][", ", ").strip("']['").split("', '")
    
    allWordDist = Counter(allWords)
    if item.name != "Overall" and common == "false":
        for word in list(allWordDist):
            if word in stopCounter:
                del allWordDist[word]

    result = tableResults(allWordDist, item.name, top)
    
    if item.name == "Overall" and common == "false":
        stopCounter = result['Words: ' + str(item.name)].tolist()
    
    return result

def td_idf(item, top):
    
    vectorizer = TfidfVectorizer(lowercase=False)
    
    matrix = vectorizer.fit_transform(item).todense()
    matrix = pd.DataFrame(matrix, columns=vectorizer.get_feature_names())
    top_words = matrix.sum(axis=0).sort_values(ascending=False)
    
    result = tableResults(top_words, item.name, top)
    result = result.round({result.columns[1]: 2})
    
    return result

def tableResults(allWordDist, name, top):
    
    wordCount = pd.Series(allWordDist).to_frame(name)
    wordCount.sort_values(by=[name], inplace = True, ascending=False)
    if (int(top)> wordCount.size):
        top = wordCount.size
    result = wordCount.head(int(top)).reset_index()
    result = result.rename(columns={result.columns[0]: 'Words: ' + str(name)})
    result = result.rename(columns={result.columns[1]: 'Count: ' + str(name)})
    
    return result
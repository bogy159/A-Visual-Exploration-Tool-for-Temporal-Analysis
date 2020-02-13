import pandas as pd
from collections import Counter
from itertools import chain

def main(item, granulation):
    
    mid = prepareData(item, granulation)
    result = frameY(mid)
    
    return result

def filterTime(item,time):
    
    result = item.copy()
    indexNames = item[item['time'] != time].index
    result.drop(indexNames , inplace=True)
    result = result.rename(columns={'text': str(time)})
    
    return result[str(time)]

def prepareData(item, granulation):
        
    wordsDF = item.loc[:, ['lemmas']]
    item = item.loc[:, ['date']]   
    item = item.sort_values('date', ascending = True)
        
    if granulation == "Year":
        item['time'] = pd.to_datetime(item.date).dt.year
    elif (granulation == "Month"):
        item['time'] = pd.to_datetime(item.date)
        item['time'] = item.time.map(lambda x: x.strftime('%Y-%m'))
        
    result = item.loc[:, ['time']]    
    
    result = pd.merge(result, wordsDF["lemmas"], left_index=True, right_index=True, how='left')
    result.rename(columns={"lemmas": 'text'}, inplace=True)
    
    return result

def frameY(item):
    start = item.rename(columns={'text': "Overall"})
    
    start = tokenCount(start["Overall"])
    
    uniqueTime = item.time.unique() 
    
    DF_list = [filterTime(item, i) for i in uniqueTime]
    
    mid = pd.concat([tokenCount(i) for i in DF_list], axis=1)
    
    result = pd.concat([start,mid], axis=1)

    return result

def tokenCount(item):
        
    allWords = chain.from_iterable(item)
    
    allWordDist = Counter(allWords)
    
    result = tableResults(allWordDist, item.name)
        
    return result

def tableResults(allWordDist, name):
    
    wordCount = pd.Series(allWordDist).to_frame(name)
    wordCount.sort_values(by=[name], inplace = True, ascending=False)
    if (20> wordCount.size):
        top = wordCount.size
    result = wordCount.head(20).reset_index()
    result = result.rename(columns={result.columns[0]: 'Words: ' + str(name)})
    result = result.rename(columns={result.columns[1]: 'Count: ' + str(name)})
    
    return result
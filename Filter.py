
# coding: utf-8

import pandas as pd 
from ast import literal_eval

def main(ID, rating, dateLow, dateUp, toYear, weekDay, IDlist):
    
    result = readCSV('static/revSamLem.csv')
    
    #result["lemmas"] = result["lemmas"].apply(literal_eval)
    
    if (ID!=''):
        result = IDFilter(result, ID)
    if (rating!=''):
        result = ratingFilter(result, rating)
    if (dateLow!=''):
        result = dateLowFilter(result, dateLow)
    if (dateUp!=''):
        result = dateUpFilter(result, dateUp)
    if (toYear!=''):
        result = toYearFilter(result, toYear)
    if (weekDay!=''):
        result = weekDayFilter(result, weekDay)
    if (IDlist!=''):
        clusters = getIDlist(IDlist)
        #print(clusters)
        result = IDlistFilter(result, clusters)
    
    return result

def readCSV(item):
    data = pd.read_csv(item, dtype={0: int}, index_col=0, encoding='latin-1')
    df = pd.DataFrame(data)
    return df

def IDFilter(result, item):
    indexNames = result[result.business_id != item].index
    result.drop(indexNames , inplace=True)
    
    return result

def ratingFilter(result, item):
    indexNames = result[result.stars != int(item)].index
    result.drop(indexNames , inplace=True)
    
    return result

def dateLowFilter(result, item):
    indexNames = result[pd.to_datetime(result.date) < pd.to_datetime(item)].index
    result.drop(indexNames , inplace=True)
    
    return result

def dateUpFilter(result, item):
    indexNames = result[pd.to_datetime(result.date) > pd.to_datetime(item)].index
    result.drop(indexNames , inplace=True)
    
    return result

def toYearFilter(result, item):
    indexNames = result[pd.to_datetime(result.date).dt.month != int(item)+1].index
    result.drop(indexNames , inplace=True)
    
    return result

def weekDayFilter(result, item):
    indexNames = result[pd.to_datetime(result.date).dt.dayofweek != int(item)].index
    result.drop(indexNames , inplace=True)
    
    return result

def getIDlist(item):
    clustercheta2 = readCSV('static/revClust.csv')

    clustercheta2 = clustercheta2.loc[:,['business_id', 'Cluster']]
    clustercheta2.set_index('business_id', inplace = True)
    clustercheta2 = clustercheta2.loc[~clustercheta2.index.duplicated(keep='first')] 
    clustercheta2.reset_index(inplace = True)

    indexNames = clustercheta2[clustercheta2.Cluster != int(item)].index
    clustercheta2.drop(indexNames , inplace=True)
    
    return clustercheta2.business_id

def IDlistFilter(result, item):    
    result = result[result['business_id'].isin(item)]
    
    return result
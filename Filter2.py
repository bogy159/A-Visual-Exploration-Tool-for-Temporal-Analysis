
# coding: utf-8

import pandas as pd 

def main(data, ID, newName, rating, dateLow, dateUp):#, toYear, weekDay, IDlist):
    
    result = data
        
    if (ID!=''):
        result = IDFilter(result, ID)
    if (newName!=''):
        result = newNameFilter(result, newName)
    if (rating!=''):
        result = ratingFilter(result, rating)
    if (dateLow!=''):
        result = dateLowFilter(result, dateLow)
    if (dateUp!=''):
        result = dateUpFilter(result, dateUp)
        
    #if (toYear!=''):
    #    result = toYearFilter(result, toYear)
    #if (weekDay!=''):
    #    result = weekDayFilter(result, weekDay)
    #if (IDlist!=''):
    #    clusters = getIDlist(IDlist)    
    #    result = IDlistFilter(result, clusters)
    
    return result

def readCSV(item):
    data = pd.read_csv(item, dtype={0: int}, index_col=0, encoding='latin-1')
    df = pd.DataFrame(data)
    return df

def IDFilter(result, item):
    #indexNames = result[result.business_id != item].index
    #result.drop(indexNames , inplace=True)
    
    result = result[result.business_id == item]
        
    return result

def newNameFilter(result, item):
    
    #indexNames = result[result.new_name != item].index
    #result.drop(indexNames , inplace=True)  
    
    result = result[result.new_name == item]  
        
    return result 
    
def ratingFilter(result, item):
    item = int(item)
    #indexNames = result[result.stars != item].index
    #result.drop(indexNames , inplace=True)
    
    result = result[result.stars == item]
    
    return result

def dateLowFilter(result, item):
    item = pd.to_datetime(item)
    #indexNames = result[pd.to_datetime(result.date) <= item].index
    #result.drop(indexNames , inplace=True)
    
    result = result[pd.to_datetime(result.date) >= item]
    
    return result

def dateUpFilter(result, item):
    item = pd.to_datetime(item)
    #indexNames = result[pd.to_datetime(result.date) >= item].index
    #result.drop(indexNames , inplace=True)
    
    result = result[pd.to_datetime(result.date) <= item]
    
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

# coding: utf-8

import pandas as pd 
import numpy as np
import time
import random
import math

from matplotlib import pyplot as plt
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as dist
from scipy.stats import zscore
from scipy import signal, stats
from scipy.signal import savgol_filter

def main(dataset, clusNum, clMethod, clMetric, trend, average, normalise, smooth):
    
    
    if clMethod == "Simple" or clMethod == "simple":
        impute = False
        proba1 = forPeriod(dataset, average, impute)
        clustercheta = pd.DataFrame(hard_clustering(proba1, plot=False))
        result = addingColumns(dataset,proba1,clustercheta)
        
    else:  
        impute = True
        proba1 = forPeriod(dataset, average, impute)

        if smooth == "True" or smooth == "true":
            proba1 = smoothing(proba1)
        if normalise == "zscore" or normalise == "Zscore":
            proba1 = proba1.apply(zscore, axis =1, result_type='broadcast')
        elif normalise == "minmax" or normalise == "Minmax":
            proba1 = normalizeMatrix(proba1)

        if trend == "true" or trend == "True":
            proba1 = trending(proba1, average)
        clustercheta = pd.DataFrame(print_clusters(proba1, clMethod, clMetric, clusNum, plot=False))
        result = addingColumns(dataset,proba1,clustercheta)
        
    return writing(result)

def reading():

    def readCSV(item):
        data = pd.read_csv(item, dtype={0: int}, index_col=0, encoding='latin-1')
        df = pd.DataFrame(data)
        return df

    restaurants = readCSV('static/revSam.csv')
    
    return restaurants

def writing(item):
    check = False
    try:
        item.to_csv('static/revClust.csv')
    except:
        check = False
        return "Unable to overwrite revClust.csv!"
    check = True
    return check

def hot_deck(dataframe) :
        dataframe = dataframe.fillna(0)
        for col in dataframe.columns :
            assert (dataframe[col].dtype == np.float64) | (dataframe[col].dtype == np.int64)
            liste_sample = dataframe[dataframe[col] != 0][col].unique()
            dataframe[col] = dataframe.apply(lambda row : random.choice(liste_sample) if row[col] == 0 else row[col],axis=1)
        return dataframe

def dendograming(item, clMethod, clMetric):
    # Here we use spearman correlation
    def my_metric(x, y):
        r = stats.pearsonr(x, y)[0]
        return 1 - r # correlation to distance: range 0 to 2

    def crossCorrelation(x, y):
        corr = signal.correlate(x, y, mode='same') 
        shift = int(np.argmax(corr) - len(x) / 2)

        sig_shift = y
        sig_shift = np.roll(sig_shift, shift)

        r = stats.pearsonr(x, sig_shift)[0]
        return 1 - r # correlation to distance: range 0 to 2
    
    if clMetric == "crossCorrelation":        
        clMetric = crossCorrelation

    #Y=dist.pdist(revMedian, 'correlation')

    Z = sch.linkage(item, method=clMethod, metric=clMetric)

    # Plot dendogram
    plt.figure(figsize=(10, 6))
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('business_id')
    plt.ylabel('distance')
    sch.dendrogram(
        Z,
        leaf_rotation=1.,  # rotates the x axis labels
        leaf_font_size=8.,  # font size for the x axis labels
    )
    
    return plt.show()

def print_clusters(timeSeries, clMethod, clMetric, k, plot):
    plt.close('all')
    
    def crossCorrelation(x, y):
        corr = signal.correlate(x, y, mode='same') 
        shift = int(np.argmax(corr) - len(x) / 2)

        sig_shift = y
        sig_shift = np.roll(sig_shift, shift)

        r = stats.pearsonr(x, sig_shift)[0]
        return 1 - r # correlation to distance: range 0 to 2
    
    if clMetric == "crossCorrelation":        
        clMetric = crossCorrelation
    
    Z =sch.linkage(timeSeries, method=clMethod, metric=clMetric)
    
    # k Number of clusters I'd like to extract
    results = sch.fcluster(Z, k, criterion='maxclust')
    
    # check the results
    s = pd.Series(results)
    clusters = s.unique()
    for c in clusters:
        cluster_indeces = s[s==c].index
        if plot:
            timeSeries.T.iloc[:,cluster_indeces].plot()
            plt.show()
    return s

def hard_clustering(timeSeries, plot):
    plt.close('all')
    
    meanAll = np.nanmean(timeSeries)    
    negativeMean = []
    positiveMean = []
    
    for i in timeSeries.iterrows():
        values = [item for item in i[1] if not math.isnan(item)]        
        half1 = values[:int(len(values)/2)]
        half2 = values[int(len(values)/2):]
        quart1 = values[:int(len(values)/4)]
        quart4 = values[-int(len(values)/4):]
        third1 = values[:int(len(values)/3)]
        third3 = values[-int(len(values)/3):]
        fift1 = values[:int(len(values)/5)]
        fift5 = values[-int(len(values)/5):]
        distance = ((np.mean(half1) + np.mean(quart1) + np.mean(third1) + np.mean(fift5))/4)-((np.mean(half2) + np.mean(quart4) + np.mean(third3) + np.mean(fift5))/4)
        if distance > 0:
            positiveMean = np.append(positiveMean, distance)
        elif distance < 0:
            negativeMean = np.append(negativeMean, distance)
    
    results = []
    for i in timeSeries.iterrows():
        values = [item for item in i[1] if not math.isnan(item)]
        
        half1 = values[:int(len(values)/2)]
        half2 = values[int(len(values)/2):]
        quart1 = values[:int(len(values)/4)]
        quart4 = values[-int(len(values)/4):]
        third1 = values[:int(len(values)/3)]
        third3 = values[-int(len(values)/3):]
        distance = ((np.mean(half1) + np.mean(quart1) + np.mean(third1) + np.mean(fift5))/4)-((np.mean(half2) + np.mean(quart4) + np.mean(third3) + np.mean(fift5))/4)
                
        if distance >= np.mean(positiveMean):
            results = np.append(results, 1)            
        elif distance <= np.mean(negativeMean):
            results = np.append(results, 2)
        elif np.mean(values) > meanAll:
            results = np.append(results, 3)
        elif np.mean(values) <= meanAll:
            results = np.append(results, 4)
        else:
            results = np.append(results, 0)
    
    # check the results
    s = pd.Series(results)
    clusters = s.unique()
    
    for c in clusters:
        cluster_indeces = s[s==c].index
        if plot:
            timeSeries.T.iloc[:,cluster_indeces].plot()
            plt.show()
    
    return s

def addingColumns(item,glavna,clustercheta):

    clustercheta2 = glavna.reset_index()
    clustercheta2.insert(1,"Cluster", clustercheta,True)
    clustercheta2 = clustercheta2.loc[:,['business_id', 'Cluster']]
    clustercheta2.set_index('business_id', inplace = True)

    item['occur'] = item.groupby('business_id')['business_id'].transform('size')
    item.set_index('business_id', inplace=True)
    item = item.loc[~item.index.duplicated(keep='first')]    
    item = item.drop(item.columns.difference(['name', 'occur', 'postal_code']), 1)
    
    glavna.reset_index(inplace = True)
    glavna = pd.melt(glavna,id_vars=['business_id'],var_name='Month', value_name='Median')
    glavna.set_index('business_id', inplace=True)
    result = pd.merge(glavna, clustercheta2, left_index=True, right_index=True, how='outer')  
    result = pd.merge(result, item, left_index=True, right_index=True, how='outer')    
    result.reset_index(inplace = True)
    result.sort_values(['occur', 'Month'], ascending=[False, True], inplace = True)
    
    return result

def forPeriod(item, average, impute):
        
    revSam3 = item
        
    revSam3['date'] = pd.to_datetime(revSam3.date)
    revSam3['date'] = revSam3.date.map(lambda x: x.strftime('%Y-%m'))
    
    if average == "median" or average == "Median":
        revMedian = revSam3.groupby(['business_id', 'date'])['stars'].median()
    elif average == "mean" or average == "Mean":
        revMedian = revSam3.groupby(['business_id', 'date'])['stars'].mean()
    revMedian = pd.DataFrame(data=revMedian)
    revMedian.rename(columns = {'stars':'median'}, inplace = True)
    revMedian = revMedian.pivot_table(index='business_id',columns='date', values={'median'},aggfunc='sum')
    revMedian.columns =  revMedian.columns.droplevel(level = 0)
    
    if impute:
        revMedian = hot_deck(revMedian)
    
    return revMedian

def trending(dataset, average):
    if average == "median" or average == "Median":
        for item in dataset.index:
            dataset.loc[ item , : ] = dataset.loc[ item , : ].rolling(12).median()
    elif average == "mean" or average == "Mean":
        for item in dataset.index:
            dataset.loc[ item , : ] = dataset.loc[ item , : ].rolling(12).mean()
    result = dataset.dropna(axis='columns')
    
    return result

def normalizeMatrix(item):
    item = item.transpose()
    result =((item-item.min())/(item.max()-item.min())*1)
    result = result.transpose()
    return result

def smoothing(item):
    window_size = 3
    if item.shape[1]<=19:
        window_size = 3
    elif (int(item.shape[1]/8)%2 == 0):
        window_size = int(item.shape[1]/8)+5
    else:
        window_size = int(item.shape[1]/8)+6
        
    polyorder = int((window_size-1)/2.5)
    
    result = item.apply(savgol_filter, args=(window_size,polyorder), axis =1, result_type='broadcast')
    
    def bound(x):
        if x > 5:
            x=5
        if x < 1:
            x=1
        return x
    
    result = result.applymap(bound)    
    
    return result
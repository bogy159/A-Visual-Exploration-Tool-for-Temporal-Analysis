
# coding: utf-8

import pandas as pd
import nltk
import re
from itertools import chain
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob
import spacy
nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])
stopwords = set(nltk.corpus.stopwords.words('english'))
newStopWords = {}


def main(item, boolean, top, granulation, getLemma, frequency, phrases):
    mid = prepareData(item, granulation)
    result = frameY(mid, boolean, top, getLemma, frequency, phrases)
    
    return result

def prepareData(item, granulation):
    
    item = item.loc[:, ['date','text']]
    item = item.sort_values('date', ascending = True)
    if granulation == "Year":
        item['time'] = pd.to_datetime(item.date).dt.year
    elif (granulation == "Month"):
        item['time'] = pd.to_datetime(item.date)
        item['time'] = item.time.map(lambda x: x.strftime('%Y-%m'))
    result = item.loc[:, ['time','text']]
    
    return result

def filterTime(item,time):
    
    result = item.copy()
    indexNames = item[item['time'] != time].index
    result.drop(indexNames , inplace=True)
    result = result.rename(columns={'text': str(time)})
    
    return result[str(time)]

def tokenCount(item, top, getLemma, phrases):
    
    global newStopWords
    if (phrases == "True" or phrases == "true"):
        allWords = item.apply(phrasesTB).tolist() 
        #Here we flatten the list
        allWords = list(chain.from_iterable(allWords))
    else:        
        tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
        allWords = item.apply(tokenizer.tokenize).tolist()        
        allWords = list(chain.from_iterable(allWords)) 
        if getLemma == "True" or getLemma == "true":
            allWords = lemmatizingSpacy(allWords, True) 
        elif getLemma == "stem" or getLemma == "Stem":
            allWords = stemmingSnowball(allWords, True)
 
    allWordExceptStopDist = nltk.FreqDist(w.lower() for w in allWords if (w.lower() not in stopwords) and (w.lower() not in newStopWords)) 

    wordCount = pd.Series(allWordExceptStopDist).to_frame(item.name)
    wordCount.sort_values(by=[item.name], inplace = True, ascending=False)
    if (int(top)> wordCount.size):
        top = wordCount.size
    result = wordCount.head(int(top)).reset_index()
    result = result.rename(columns={result.columns[0]: 'Words: ' + str(item.name)})
    result = result.rename(columns={result.columns[1]: 'Count: ' + str(item.name)})
    
    return result

def tf_idf(item, top, getLemma, phrases, common):
    
    if (phrases == "True" or phrases == "true"):
        mid = wWeighting(item, phrases, common)    
    elif getLemma == "True" or getLemma == "true":
        item = item.apply(removeSWSentance, args=[True])
        item = item.apply(lemmatizingSpacy, args=[False])
        mid = wWeighting(item, phrases, common)    
    elif getLemma == "stem" or getLemma == "Stem":
        item = item.apply(removeSWSentance, args=[True])
        item = item.apply(stemmingSnowball, args=[False])
        mid = wWeighting(item, phrases, common)  
    else:
        item = item.apply(removeSWSentance, args=[False])
        mid = wWeighting(item, phrases, common)    
        
    wordCount = pd.Series(mid).to_frame(item.name)
    wordCount.sort_values(by=[item.name], inplace = True, ascending=False)    
    
    if (int(top)> wordCount.size):
        top = wordCount.size
    result = wordCount.head(int(top)).reset_index()
    result = result.rename(columns={result.columns[0]: 'Words: ' + str(item.name)})
    result = result.rename(columns={result.columns[1]: 'Count: ' + str(item.name)})
    result = result.round({result.columns[1]: 2})
    
    return result

def frameY(item, common, top, getLemma, frequency, phrases):
    
    global newStopWords
    newStopWords = {}
    start = item.rename(columns={'text': "Overall"})
    
    if frequency == "True" or frequency == "true":
        start = tf_idf(start["Overall"], top, getLemma, phrases, common)
    else:
        start = tokenCount(start["Overall"], top, getLemma, phrases)
    
    if common == "False" or common == "false":
        newStopWords = set(start["Words: Overall"].str.lower())
    
    uniqueTime = item.time.unique()   
    DF_list = [filterTime(item, i) for i in uniqueTime]
    if frequency == "True" or frequency == "true":
        result = pd.concat([tf_idf(i, top, getLemma, phrases, common) for i in DF_list], axis=1)
    else:
        result = pd.concat([tokenCount(i, top, getLemma, phrases) for i in DF_list], axis=1)
    
    result = pd.concat([start,result], axis=1)
    
    result.index+=1
    result.index.names = ['Place']

    return result

def lemmatizingSpacy(item, isList):
    
    document = spacy.tokens.Doc(nlp.vocab, words=item)
    if(isList):        
        result = [w.lemma_ for w in document]
    else:
        result = " ".join([w.lemma_ for w in document])
    return result

def stemmingSnowball(item, isList):
    
    ps = nltk.stem.SnowballStemmer('english')        
    if(isList):        
        result = [ps.stem(w) for w in item]
    else:
        result = " ".join([ps.stem(w) for w in item])
    return result

def removeSWSentance(sentence, isList):
    global newStopWords
    
    tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
    allWords = tokenizer.tokenize(sentence)
    
    if(isList):
        result = [word for word in allWords if (word.lower() not in stopwords) and (word.lower() not in newStopWords) and (word != "-PRON-")]
    else:
        result = " ".join([word for word in allWords if (word.lower() not in stopwords) and (word.lower() not in newStopWords) and (word != "-PRON-")])
    
    return result

def wWeighting(item, phrases, common):
    
    if (phrases == "True" or phrases == "true"):
        if (common == "False" or common == "false"):
            vectorizer = TfidfVectorizer(ngram_range=(2,3), stop_words = 'english', preprocessor=remove_stop_phrases)
        else:
            vectorizer = TfidfVectorizer(ngram_range=(2,3), stop_words = 'english')
    else:
        vectorizer = TfidfVectorizer()
    
    matrix = vectorizer.fit_transform(item).todense()
    # transform the matrix to a pandas df
    matrix = pd.DataFrame(matrix, columns=vectorizer.get_feature_names())
    # sum over each document (axis=0)
    top_words = matrix.sum(axis=0).sort_values(ascending=False)
    
    return top_words

def remove_stop_phrases(doc):    
    global newStopWords
    stop_phrases = newStopWords
    
    for phrase in stop_phrases:
        doc = re.sub(phrase, "", doc, flags=re.IGNORECASE)
    return doc

def phrasesTB(text):
    value = TextBlob(text).noun_phrases
    return value
from textblob import TextBlob
import pandas as pd

def main(item):
    item['polarity'] = item.text.apply(detect_polarity)
    
    positive = item.loc[item['polarity'] == "Positive"]
    negative = item.loc[item['polarity']=="Negative"]
    neutral = item.loc[item['polarity']=="Neutral"]
    positive = positive.drop(columns="polarity")
    negative = negative.drop(columns="polarity")
    neutral = neutral.drop(columns="polarity")
    
    return negative, neutral, positive 

def detect_polarity(text):
    value = TextBlob(text).sentiment.polarity
    if value > 0.3:
        return "Positive"
    elif value < -0.3:
        return "Negative"
    else:
        return "Neutral"
import pandas as pd
from collections import Counter
from itertools import chain


def main(series):
    
    fullList = chain.from_iterable(series)
    counted = Counter(fullList)
    words = counted.most_common(10)
    result = ""
    for x in words:
        result = result + " '" + x[0] + "'"
    result = "Top words:" + result
    
    return result
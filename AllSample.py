import pandas as pd 

def main():
    data = readCSV('static/revSam.csv')
    
    data["newID"] = data["name"].map(str) +" - "+ data["postal_code"].map(str)
    data.drop(data.columns[[0,2,3,4,5,6]],axis=1,inplace=True)
    
    data = pd.DataFrame(data.groupby(data.columns.tolist(),as_index=True).size()).reset_index()
    data = data.rename(columns={0: "count"})
    data = data.pivot(index='newID', columns='stars', values='count').reset_index()
    data = data.rename(columns={1: "1_star", 2: "2_star", 3: "3_star", 4: "4_star", 5: "5_star", })
    
    data['total'] = data[['1_star', '2_star', '3_star', '4_star', '5_star']].sum(axis=1) 
    data = data.sort_values(by=['total'], ascending=False)
    data.drop("total",axis=1,inplace=True)
    
    return data

def readCSV(item):
    data = pd.read_csv(item, dtype={0: int}, index_col=0, encoding='latin-1')
    df = pd.DataFrame(data)
    return df
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re

import pandas as pd
x=str(input('What type of separation is there in the file such as (,) or (;)'))
y=str(input('Address of the training file'))
train = pd.read_csv(y,sep=x)

import statsmodels.api as sm
def backward_elimination(data, target,significance_level = 0.05):
    features = data.columns.tolist()
    while(len(features)>0):
        features_with_constant = sm.add_constant(data[features])
        p_values = sm.OLS(target, features_with_constant.astype(float)).fit().pvalues[1:]
        max_p_value = p_values.max()
        if(max_p_value >= significance_level):
            excluded_feature = p_values.idxmax()
            features.remove(excluded_feature)
        else:
            break 
    return features
def training(train):
    x=[]
    for i in train.columns:
        if re.search('date', i, re.IGNORECASE):
            x.append(i)
    columns=x

    date=pd.DataFrame()
    string=train.select_dtypes(include='object')
    string.drop(columns=columns,inplace=True)
    integer=train.select_dtypes(include=['int','float'])
    date = pd.DataFrame(train, columns=columns)

    for i in date:
        date[i]= pd.to_datetime(date[i],yearfirst=True) 

    date1=pd.DataFrame(date,columns=columns)
    for i in date1:
        x=[]
        for j in date1[i]:
            x.append(int(j.strftime('%Y')))
        date[i+'_year']=x
    for i in date1:
        x=[]
        for j in date1[i]:
            x.append(int(j.strftime('%m')))
        date[i+'_month']=x
    for i in date1:
        x=[]
        for j in date1[i]:
            x.append(int(j.strftime('%d')))
        date[i+'_day']=x
    for i in date1:
        x=[]
        for j in date1[i]:
            x.append(int(j.strftime('%H')))
        date[i+'_hour']=x
    for i in date1:
        x=[]
        for j in date1[i]:
            x.append(int(j.strftime('%M')))
        date[i+'_minutes']=x
    for i in date1:
        x=[]
        for j in date1[i]:
            x.append(int(j.strftime('%S')))
        date[i+'_second']=x
    date.drop(columns=columns,inplace=True)
    del date1
    del x
    del y


    for i in integer.columns:
        integer[i].fillna(integer[i].mean(),inplace=True)

    string_1=string

    x=[]
    for i in string_1.columns:
        string_1[i] = string_1[i].str.replace(r'\D', '')
        try:
            string_1[i] = string_1[i].astype('int')
        except:
            x.append(i)
    string_1.drop(columns=x,inplace=True)
    string=pd.DataFrame(train,columns=x)

    for i in string.columns:
        string[i].fillna(string[i].mode().iloc[0],inplace=True)

    for i in string.columns:
        x=[]
        arr=string[i].unique()
        n=string[i].nunique()
        for j in string[i]:
            for k in range(0,n):
                if(j==arr[k] ):
                    x.append(k)
                else:
                    continue
        string[i]=x
    del x
    del i
    del train

    train = integer.join(string, how='inner')
    train = train.join(string_1, how='inner')
    train = train.join(date, how='inner')

    del integer
    del string_1
    del string
    del date

    x=str(input('Which is the column you want to predict?'))
    x_train=train.drop(columns=x)
    y_train=train[x]
    columns=backward_elimination(x_train,y_train)
    x_train = pd.DataFrame(x_train, columns=columns)
    return x_train,y_train

x_train,y_train = training(train)


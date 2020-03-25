#author@https://github.com/ekshusingh

import numpy as np
 
import pandas as pd
 
import re

import statsmodels.api as sm

from sklearn.ensemble import RandomForestClassifier

from sklearn import svm

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

def correlation(x_train):
    
    corr = x_train.corr()
    
    columns = np.full((corr.shape[0],), True, dtype=bool)
    
    for i in range(corr.shape[0]):
        
        for j in range(i+1, corr.shape[0]):
            
            if corr.iloc[i,j] >= 0.5:
                
                if columns[j]:
                    
                    columns[j] = False
                    
    selected_columns = x_train.columns[columns]
    
    x_train = x_train[selected_columns]
    
    return x_train


def date_pre(date,columns):
    
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
    
    return date




def trainingandtesting(x_train,test):
    
    
    predict_column='status_group'
    
    train=x_train.drop(columns=predict_column)
    
    y_train=pd.DataFrame(x_train,columns=[predict_column])
    
    for i in y_train.columns:
        
        x=[]
        
        arr=y_train[i].unique()
    
        
        n=y_train[i].nunique()
        
        for j in y_train[i]:
            
            for k in range(0,n):
                
                if(j==arr[k] ):
                    x.append(k)
                    
                else:
                    
                    continue
    
        y_train[i]=x
    
    x=[]
    
    for i in train.columns:
        
        if re.search('date', i, re.IGNORECASE):
            
            x.append(i)
            
    columns=x
    
    date_train=pd.DataFrame()
    
    date_test=pd.DataFrame()
    
    string_train=train.select_dtypes(include='object')
    
    string_test= train.select_dtypes(include='object')
    
    string_train.drop(columns=columns,inplace=True)
    
    string_test.drop(columns=columns,inplace=True)
    
    integer_train=train.select_dtypes(include=['int','float','int64'])
    
    integer_test=test.select_dtypes(include=['int','float','int64'])
    
    date_train = pd.DataFrame(train, columns=columns)
    
    date_test= pd.DataFrame(test, columns=columns)
    
    date_train=date_pre(date_train,columns)
    
    date_test=date_pre(date_test,columns)
    
    string_train_number,string_train_1,string_test_number,string_test_1=string_train_pre(string_train,train,string_test,test)     
    
    for i in integer_train.columns:
        
        integer_train[i].fillna(integer_train[i].mean(),inplace=True)
        integer_test[i].fillna(integer_test[i].mean(),inplace=True)
        
    del train
    
    train = integer_train.join(string_train_number, how='inner')
    
    train = train.join(string_train_1, how='inner')
    
    train = train.join(date_train, how='inner')
    
    del test
    
    test = integer_test.join(string_test_number, how='inner')
    
    test = test.join(string_test_1, how='inner')
    
    test = test.join(date_test, how='inner')
    
    columns=backward_elimination(train,y_train)
    
    train = pd.DataFrame(train, columns=columns)
    
    train=correlation(train)
    
    test= pd.DataFrame(test,columns=train.columns)
    
    return train,y_train,test
    
#author@https://github.com/ekshusingh

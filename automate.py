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
            
            if corr.iloc[i,j] >= 0.7:
                
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


def string_train_pre(string_train,train,string_test,test):
    
     string_train_1=string_train
    
     string_test_1=string_test
    
     x=[]
    
     for i in string_train_1.columns:
        
         string_train_1[i] = string_train_1[i].str.replace(r'\D', '')
        
         string_test_1[i] = string_test_1[i].str.replace(r'\D', '')
        
         try:
            
             string_train_1[i] = string_train_1[i].astype('int')
            
             string_test_1[i] = string_test_1[i].astype('int')
            
         except:
            
             x.append(i)
            
     string_train_1.drop(columns=x,inplace=True)
    
     string_test_1.drop(columns=x,inplace=True)
    
     string_train=pd.DataFrame(train,columns=x)
    
     string_test=pd.DataFrame(test,columns=x)

     for i in string_train.columns:
        
        string_train[i].fillna(string_train[i].mode().iloc[0],inplace=True)
        
        string_test[i].fillna(string_train[i].mode().iloc[0],inplace=True)
  
     for i in string_train.columns:
        
        x_train=[]
        
        x_test=[]
        
        arr_train=string_train[i].unique()
                
        n=string_train[i].nunique()
        
        for j in string_train[i]:
            
            for k in range(0,n):
                
                if(j==arr_train[k] ):
                    
                    x_train.append(k)
                    
                else:
                    
                    continue
                
        string_train[i]=x_train
        
        arr_test=string_test[i].unique()
        
        arr_left=list(set(arr_test).difference(arr_train))
        
        number_of_not=[]
        
        number=n
        
        for l in arr_left:
            number=number+1
            number_of_not.append(number)
        
        arr_left = dict(zip(number_of_not, arr_left)) 
        
         
        for j in string_test[i]:
            
            for k in range(0,n):
                
                    if(j==arr_train[k] ):
                    
                        x_test.append(k)
                
                    else:
                        
                        for key,value in arr_left.items():
                            
                            if(j==value):
                                
                                x_test.append(int(key))
                                
                                break
                        else:
                            
                            continue
                        
                        break
        
        string_test[i]=x_test
        
     return string_train,string_train_1,string_test,string_test_1

def trainingandtesting(x_train,test):
    
    
    predict_column=str(input('enter which column to predict: '))
    
    train=x_train.drop(columns=predict_column)
    
    y_train=pd.DataFrame(x_train,columns=[predict_column])
    
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
    
    integer_train=train.select_dtypes(include=['int64','float'])
    
    integer_test=test.select_dtypes(include=['int64','float'])
    
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
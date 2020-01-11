#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 11 05:42:55 2020

@author: Mr.Gosselin
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

# Assuming pre-processing of the data has already been done. Specifically, there should be only x1 to x26 plus a y columns and no error in the dataset.

def predicting():
    data = pd.read_csv(r'mistplay-challenge.csv')
    
    # Dropping duplicate, unuseful, or undesired columns
    data = data.drop(columns = ['x1', 'x15'], axis = 1)
    # Fixing datatypes
    for i in ['x5', 'x8', 'x16', 'x17']:
        data[i] = data[i].astype('int')
    data['x23'] = data['x23'].replace('TRUE', True).replace('FALSE', False).astype('bool')
    
    # Creating dummy variables for categorical variables
    numerical_col = ['x6', 'x7', 'x9', 'x11', 'x12','x13', 'x14', 'x17']
    categorical_col = data.columns[~data.columns.isin(numerical_col)].tolist()
    response_col = data.y
    data_cat = data[categorical_col]
    data = data.drop(data_cat.columns.tolist(), axis=1)
    data_cat = pd.concat([pd.get_dummies(data_cat[col]) for col in data_cat], axis=1, keys=data_cat.columns)
    data_cat.columns = [col[1] for col in data_cat.columns]
    data = pd.concat([data_cat, data, response_col], axis=1)
    
    nonzero = data[data['y'] != 0]
    zero = data[data['y'] == 0].sample(n=len(nonzero), random_state = 12)
    df = pd.concat([nonzero, zero], axis = 0)
    extra_data = data[~data.index.isin(df.index)]
    
    # Defining training and test datasets
    y = df['y']; y_extra = extra_data['y']
    X = df.drop(columns = 'y'); X_extra = extra_data.drop(columns = 'y')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=12) # 31/148 non-zeros in y_test
    X_test = pd.concat([X_test, X_extra], axis=0); y_test = pd.concat([y_test, y_extra], axis=0)
    
    # Scaling non-categorical variables
    scaler = StandardScaler()
    data.apply(lambda x: x.nunique()) # Not knowing column representation, we'll scale only columns with more than 5 unique values
    scaler.fit(X_train.loc[:, numerical_col]) 
    X_train.loc[:, numerical_col] = scaler.transform(X_train.loc[:, numerical_col])
    X_test.loc[:, numerical_col] = scaler.transform(X_test.loc[:, numerical_col])
    
    # Baseline model
    len(y_test)
    baseline = [0 for i in range(len(y_test))]
    baseline_error = np.mean(abs(baseline - y_test))
    
    # # Random forest
    rf = RandomForestRegressor(n_estimators = 1000, random_state=12)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    error_rf = np.mean(abs(rf_pred - y_test)) # Still worse than baseline error
    
    # Print model outcome
    if (baseline_error < error_rf):
        print('Our model underperforms our baseline error.')
        print('Baseline error: ', baseline_error)
        print('model error: ', error_rf)
    else:
        print('Our model outperforms the baseline error.')
        print('Baseline error: ', baseline_error)
        print('model error: ', error_rf)
    
    rf_pred = pd.DataFrame(rf_pred)
    # Output predicted y-values
    rf_pred.to_csv(r'predictedYvalues.csv')
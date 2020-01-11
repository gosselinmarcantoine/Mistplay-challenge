#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 09:42:32 2020

@author: Mr.Gosselin
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt

# Load in the data
data = pd.read_csv(r'DataScienceChallenge.csv')

# Data Exploration
data.shape # expected 26 columns but got 29
#data.info() # data types to review: x5 (int), x6 (str), x8 (int), x16 (int), x17 (int), x23 (bool)
# x6 is most likely birth year. We'll leave as int for now.

data['Unnamed: 28'].value_counts()
data[data['Unnamed: 28'] == 3.0]
data.loc[data['Unnamed: 28'] == 3.0, 'x26'] = 'Easy as 1 2 3'
data[data['Unnamed: 27'].notnull()][['x25', 'Unnamed: 27']] # Duplicate column

# Drop empty, unique and duplicate columns
data.apply(lambda x: x.nunique(), axis=0)
data.apply(lambda x: sum(x.isnull()),axis=0)

data = data.drop(columns = ['x1', 'x15', 'Unnamed: 27', 'Unnamed: 28'], axis = 1)

# Fixing dtypes
data.x5.value_counts() 
data[data['x5'] == 'US']
# Remove record with error (Fixing the error without business/data understanding is not recommended)
data = data[data['x5'] != 'US']

for i in ['x5', 'x8', 'x16', 'x17']:
    data[i] = data[i].astype('int')

# Converting column x23 to Bool
data['x23'] = data['x23'].replace('TRUE', True).replace('FALSE', False).astype('bool')

# Keeping relevant information for X24, X25, X26 (prevent model from overfitting)
#data.loc[~data.x24.isin(data.x24.value_counts()[data.x24.value_counts() > 100].index.tolist()), 'x24'] = 'nan'
#data.loc[~data.x25.isin(data.x25.value_counts()[data.x25.value_counts() > 100].index.tolist()), 'x25'] = 'nan'
#data.loc[~data.x26.isin(data.x26.value_counts()[data.x26.value_counts() > 100].index.tolist()), 'x26'] = 'nan'

# Creating dummy variables
numerical_col = ['x6', 'x7', 'x9', 'x11', 'x12','x13', 'x14', 'x17']
categorical_col = data.columns[~data.columns.isin(numerical_col)].tolist()
response_col = data.y
data_cat = data[categorical_col]
data = data.drop(data_cat.columns.tolist(), axis=1)
data_cat = pd.concat([pd.get_dummies(data_cat[col]) for col in data_cat], axis=1, keys=data_cat.columns)
data_cat.columns = [col[1] for col in data_cat.columns]
data = pd.concat([data_cat, data, response_col], axis=1)

# Investigating target (y) column
print('Number of rows with non-zero target values: ', len(data[data['y'] != 0]))

# We want our model to do better than if we simply guessed y = 0 for all records.

# Balancing dataset between non-zero to zero values in y-column
nonzero = data[data['y'] != 0]
zero = data[data['y'] == 0].sample(n=149, random_state = 12)
df = pd.concat([nonzero, zero], axis = 0)
extra_data = data[~data.index.isin(df.index)]

# Defining training and test datasets
y = df['y']; y_extra = extra_data['y']
X = df.drop(columns = 'y'); X_extra = extra_data.drop(columns = 'y')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=12) # 31/148 non-zeros in y_test
X_test = pd.concat([X_test, X_extra], axis=0); y_test = pd.concat([y_test, y_extra], axis=0)
#print(len(X_train), len(y_train), len(X_test), len(y_test))

# Scaling
scaler = StandardScaler()
data.apply(lambda x: x.nunique()) # Not knowing column representation, we'll scale only columns with more than 5 unique values
scaler.fit(X_train.loc[:, numerical_col]) 
X_train.loc[:, numerical_col] = scaler.transform(X_train.loc[:, numerical_col])
X_test.loc[:, numerical_col] = scaler.transform(X_test.loc[:, numerical_col])

# Baseline model
len(y_test)
baseline = [0 for i in range(len(y_test))]
baseline_error = np.mean(abs(baseline - y_test))
print('baseline error: ', baseline_error)

# PLS
pls2 = PLSRegression(5, scale=False)
pls2.fit(X_train, y_train)
# Quick linear regression 
Y_pred = pls2.predict(X_test)
score = pls2.score(X_test, y_test) 
lin_error = np.mean(abs([item for sublist in Y_pred.tolist() for item in sublist] - y_test)) # Bad. Perhaps a random forest method would yield better results.
print('lin error: ', lin_error)

# Random forest
rf = RandomForestRegressor(n_estimators = 1000, random_state=12)
rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)
print('rf error: ', np.mean(abs(rf_pred - y_test))) # Still worse than baseline error

# Random forest without balancing training dataset
X_rf = data.drop(columns = 'y'); y_rf = data.y
X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(X_rf, y_rf, test_size=.2, random_state=12)

pca = PCA(.75)
pca.fit(X_train)
pca.n_components_

X_train_rf = pca.transform(X_train_rf)
X_test_rf = pca.transform(X_test_rf)

rf2 = RandomForestRegressor(n_estimators = 1000, random_state=13)
rf2.fit(X_train_rf, y_train_rf)

rf_pred2 = rf2.predict(X_test_rf)
print('rf2 error: ', np.mean(abs(rf_pred2 - y_test_rf)))

# What's next: Random forest seems to be a good model for what we are trying to do. I feel like pre-processing the columns better could give us better results.
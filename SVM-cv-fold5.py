import os
import scipy.io as sio
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import mean_squared_error, r2_score 
from sklearn.preprocessing import StandardScaler 
from sklearn.metrics import confusion_matrix 
from torchsummary import summary
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.onnx

import numpy as np
import matplotlib.pyplot as plt
from eli5.sklearn import PermutationImportance
from eli5 import show_weights
import streamlit as st
import streamlit.components.v1 as components
from IPython.display import display
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler

from sklearn import svm
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

data_train = sio.loadmat("data_train (1).mat")
label_train = sio.loadmat("label_train(2).mat")

data_train = [[element for element in upperElement] for upperElement in data_train['data_train']]
label_train = [[element for element in upperElement] for upperElement in label_train['label_train']]

data_col = [str(i + 1) for i in range(len(data_train[0]))]
classes = [-1, 1]
NUM_INPUT = len(data_col)

#scaler = StandardScaler()
#X = pd.DataFrame(scaler.fit_transform(data_train), columns=data_col)
X = pd.DataFrame(data_train, columns=data_col)
y = pd.DataFrame(label_train, columns = ['output'])

#print(X.describe())
bins = 2
X['bin'] = pd.cut(y.values.ravel(), bins, labels=False)

# Prepare cross validation splits
splits = 5
skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=2022)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

fold_idx = dict()

for i, (train_idx, val_idx) in enumerate(skf.split(X.index, y)):
    fold_idx[i] = (train_idx, val_idx)

fold_results = dict()



avg_score = []
cm = []
for fold_num, (train_idx, val_idx) in fold_idx.items():
    print(f'fold: {fold_num + 1}')
    X_train = X.iloc[train_idx].copy()
    y_train = y.iloc[train_idx].copy()
    X_train = X_train.drop(['bin'], axis=1)
    #y_train.loc[y_train.output == -1, 'output'] = 0
    
    X_test = X.iloc[val_idx].copy()
    y_test = y.iloc[val_idx].copy()
    
    X_test = X_test.drop(['bin'], axis=1)
    
    # Standardization
    #scaler = StandardScaler()
    #X_train = scaler.fit_transform(X_train.to_numpy())
    #X_test = scaler.transform(X_test.to_numpy())
    
    # Convert training data to torch tensor
    #train_X = torch.FloatTensor(X_train.to_numpy()).to(device) 
    #test_X = torch.FloatTensor(X_test.to_numpy()).to(device) 
    #y_train = torch.LongTensor(y_train.to_numpy()).squeeze().to(device) 

    # instantiate classifier with default hyperparameters
    # RBF kernel by default
    #svc = SVC(C=10, kernel="rbf") 
    svc = SVC(C=5, gamma='scale', kernel="rbf") 


    # fit classifier to training set
    svc.fit(X_train,y_train.values.ravel())

    y_pred=svc.predict(X_test)

    print('Model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))
    avg_score.append(accuracy_score(y_test, y_pred))

    cm.append(confusion_matrix(y_test, y_pred))
    #print('\n Confusion matrix ') 
    #print(mat)
    #print('\n')
print(f'Average accuracy is: {np.mean(avg_score)}, std deviation: {np.std(avg_score)}')
matrix_sum = np.zeros((2, 2))

for cmatrix in cm:
    matrix_sum += cmatrix
print(matrix_sum)
#svc = SVC(C=5, gamma='scale', kernel="rbf") 
#X = X.drop(['bin'], axis=1)

#sc = cross_val_score(SVC(C=5, gamma='scale', kernel="rbf"), X, y.values.ravel(), scoring='accuracy', cv=StratifiedKFold(n_splits=splits, shuffle=True, random_state=2022))
#print(sc)
#print(f'Average accuracy is: {sc.mean()}, std deviation: {sc.std()}')
#print(sc.test_score)

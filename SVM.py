from sklearn import svm
from sklearn.svm import SVC

import os
import scipy.io as sio
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import mean_squared_error, r2_score 
from sklearn.preprocessing import StandardScaler 
from sklearn.metrics import confusion_matrix 
from torchsummary import summary

import numpy as np
import matplotlib.pyplot as plt

import eli5
from eli5.sklearn import PermutationImportance
from eli5 import show_weights
import streamlit as st
import streamlit.components.v1 as components

data_train = sio.loadmat("data_train (1).mat")
label_train = sio.loadmat("label_train(2).mat")

data_train = [[element for element in upperElement] for upperElement in data_train['data_train']]
label_train = [[element for element in upperElement] for upperElement in label_train['label_train']]

data_col = [i + 1 for i in range(len(data_train[0]))]
classes = [-1, 1]
NUM_INPUT = len(data_col)

X = pd.DataFrame(data_train, columns=data_col)
y = pd.DataFrame(label_train, columns = ['output']).values.ravel()

#y.to_csv('y.csv',header=None,index=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

def make_meshgrid(x, y, h=.02):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy
def plot_contours(ax, clf, xx, yy, **params):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out

# instantiate classifier with default hyperparameters
# RBF kernel by default
svc = SVC(C=10, kernel="rbf") 
#svc = SVC(C=4, gamma=0.08, kernel="rbf") 

# fit classifier to training set
svc.fit(X_train,y_train)

y_pred=svc.predict(X_test)

model_plt = SVC(C=10, kernel="rbf") 
clf = model_plt.fit(X_train.iloc[:, 1:3],y_train)

fig, ax = plt.subplots()
# title for the plots
title = ('Decision surface of SVM using Gaussian kernel function')
# Set-up grid for plotting.
X0, X1 = X.iloc[:, 1], X.iloc[:, 2]
xx, yy = make_meshgrid(X0, X1)

plot_contours(ax, clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors="k")
print(y)
ax.set_ylabel("x2")
ax.set_xlabel("x1")
ax.set_xticks(())
ax.set_yticks(())
ax.set_title(title)
plt.show()

#print(f'X0: {X.iloc[:, 1].to_list()}')
# make predictions on test set

# compute and print accuracy score
print('Model accuracy score with default hyperparameters: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))

mat = confusion_matrix(y_test, y_pred)
print('\n Confusion matrix ') 
print(mat)
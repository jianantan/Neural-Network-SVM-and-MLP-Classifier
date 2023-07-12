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
import torch.optim as optim

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
from sklearn.model_selection import cross_val_score
from skorch import NeuralNetRegressor, NeuralNetClassifier


NUM_EPOCHS = 2000
NUM_HIDDEN1 = 15 #4

NUM_HIDDEN2 = 470 #3
NUM_HIDDEN3 = 1 #3
NUM_HIDDEN4 = 1 #3

NUM_OUTPUT = 2

LR = 0.03 #0.01

data_train = sio.loadmat("data_train (1).mat")
label_train = sio.loadmat("label_train(2).mat")

data_train = [[element for element in upperElement] for upperElement in data_train['data_train']]
label_train = [[element for element in upperElement] for upperElement in label_train['label_train']]

data_col = [i + 1 for i in range(len(data_train[0]))]
classes = [-1, 1]
NUM_INPUT = len(data_col)

X = pd.DataFrame(data_train, columns=data_col)
y = pd.DataFrame(label_train, columns = ['output'])

print(X.shape)
print(y.shape)
bins = 2
X['bin'] = pd.cut(y.values.ravel(), bins, labels=False)

# Prepare cross validation splits
splits = 5
skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=2022)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

fold_idx = dict()

for i, (train_idx, val_idx) in enumerate(skf.split(X.index, X['bin'])):
    fold_idx[i] = (train_idx, val_idx)

fold_results = dict()

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    #random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

class MLP(nn.Module): 
  def __init__(self, numInput, num_hidden_1=NUM_HIDDEN1, num_hidden_2=NUM_HIDDEN2): 
    super(MLP, self).__init__() 
    self.fc1 = nn.Linear(numInput , num_hidden_1)  # layer 1 
    self.fc2 = nn.Linear(num_hidden_1 , NUM_OUTPUT)         # layer 2
    #self.fc3 = nn.Linear(num_hidden_2, NUM_OUTPUT)  
    #self.fc4 = nn.Linear(NUM_HIDDEN3, NUM_HIDDEN4)
    #self.fc5 = nn.Linear(NUM_HIDDEN4, NUM_OUTPUT)
    self.relu = nn.ReLU()
    self.leakyrelu = nn.LeakyReLU()
    self.sigmoid = nn.Sigmoid()
     
  def forward(self,x): 
    #x = self.fc1(x)
    x = self.relu(self.fc1(x)) 
    #x = self.fc1(x)
    #x = torch.tanh(self.fc2(x)) 
    #x = self.fc4(x)
    #x = self.fc5(x)
    #x = self.relu(self.fc2(x))
    x = self.fc2(x)
    return x

# make random numbers reproducible
#torch.manual_seed(0)
set_seed(0)

avg_score = []
for fold_num, (train_idx, val_idx) in fold_idx.items():
    print(f'fold: {fold_num + 1}')
    X_train = X.iloc[train_idx].copy()
    y_train = y.iloc[train_idx].copy()
    X_train = X_train.drop(['bin'], axis=1)
    y_train.loc[y_train.output == -1, 'output'] = 0
    
    X_test = X.iloc[val_idx].copy()
    y_test = y.iloc[val_idx].copy()
    
    X_test = X_test.drop(['bin'], axis=1)

    # Standardization
    #scaler = StandardScaler()
    #X_train = scaler.fit_transform(X_train.to_numpy())
    #X_test = scaler.transform(X_test.to_numpy())
    
    # Convert training data to torch tensor
    train_X = torch.FloatTensor(X_train.to_numpy()).to(device) 
    test_X = torch.FloatTensor(X_test.to_numpy()).to(device) 
    y_train = torch.LongTensor(y_train.to_numpy()).squeeze().to(device) 
    
    #print(y_train)
    #y_test = torch.FloatTensor(y_test.values)
    #BATCH_SIZE = len(train_X)
    #datasets = torch.utils.data.TensorDataset(train_X, y_train)
    #train_iter = torch.utils.data.DataLoader(datasets, batch_size=BATCH_SIZE, shuffle=True)
    
    model = MLP(NUM_INPUT, NUM_HIDDEN1, NUM_HIDDEN2).to(device) 
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LR)
    
    for epoch in range(NUM_EPOCHS): 
        #for x_iter, y_iter in train_iter:
        optimizer.zero_grad() 
        #print(x)
        #print(y)
        output = model(train_X) 
        #print(f'y_train is {y_train.shape}, output is {output.shape}')
        loss = criterion(output, y_train) 
        loss.backward() 
        optimizer.step() 

        if epoch % 500 == 0: 
            print(f"number of epoch {epoch}, loss = {loss.item()} ") 
            #net.zero_grad() 

    predict_out = model(test_X)
    _, y_pred = torch.max(predict_out, 1)
    y_pred = [classes[y_pred.numpy()[i]] for i in range(len(y_pred))]
    y_test = y_test.to_numpy().squeeze()
    print('Model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))
    avg_score.append(accuracy_score(y_test, y_pred))

    mat = confusion_matrix(y_test, y_pred)
    print('\n Confusion matrix ') 
    print(mat)
    print('\n')
print(f'Average accuracy is: {np.mean(avg_score)}, std deviation: {np.std(avg_score)}')


"""
model = MLP(NUM_INPUT, NUM_HIDDEN1, NUM_HIDDEN2).to(device) 
net = NeuralNetClassifier(
    module=MLP,
    module__numInput=NUM_INPUT,
    module__num_hidden_1 = NUM_HIDDEN1,
    module__num_hidden_2 = NUM_HIDDEN2,
    criterion=nn.CrossEntropyLoss,
    optimizer=optim.SGD, 
    lr = LR,
    max_epochs = NUM_EPOCHS
)

X = X.drop(['bin'], axis=1)
y.loc[y.output == -1, 'output'] = 0

sc = cross_val_score(net, 
torch.FloatTensor(X.to_numpy()).to(device), 
torch.LongTensor(y.to_numpy()).squeeze().to(device), 
scoring='accuracy', 
cv=StratifiedKFold(n_splits=splits, shuffle=True, random_state=2022)
)

print(sc)
print(f'Average accuracy is: {sc.mean()}, std deviation: {sc.std()}')
"""
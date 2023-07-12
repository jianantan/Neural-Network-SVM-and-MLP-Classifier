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


NUM_EPOCHS = 500
NUM_HIDDEN1 = 470 #4
NUM_HIDDEN2 = 100 #3
NUM_HIDDEN3 = 1 #3
NUM_HIDDEN4 = 1 #3

NUM_OUTPUT = 2
#BATCH_SIZE = 330
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
#y.to_csv('y.csv',header=None,index=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
y_train.loc[y.output == -1, 'output'] = 0
#y_test.to_csv('y_test.csv',header=None,index=True)
#print(f'y_train.dtype = {y_train.dtype}')
plt.hist(y_train)
plt.xticks([-1, 1]), plt.title(' Training labels distribution')
plt.figure()
plt.hist(y_test)
plt.xticks([-1, 1]), plt.title(' Test labels distribution')
#plt.show()

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    #random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

class MLP(nn.Module): 
  def __init__(self, numInput): 
    super(MLP, self).__init__() 
    self.fc1 = nn.Linear( numInput , NUM_HIDDEN1 )  # layer 1 
    #self.fc2 = nn.Linear(NUM_HIDDEN1 , NUM_HIDDEN2)         # layer 2
    self.fc3 = nn.Linear(NUM_HIDDEN1, NUM_OUTPUT)  
    #self.fc4 = nn.Linear(NUM_HIDDEN3, NUM_HIDDEN4)
    #self.fc5 = nn.Linear(NUM_HIDDEN4, NUM_OUTPUT)
     
  def forward(self,x): 
    #x = self.fc1(x)
    x = torch.relu(self.fc1(x)) 
    #x = self.fc1(x)
    #x = torch.tanh(self.fc2(x)) 
    #x = self.fc4(x)
    #x = self.fc5(x)
    #x = self.fc2(x)
    x = self.fc3(x)
    return x

# make random numbers reproducible
#torch.manual_seed(0)
set_seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
#print('Training on', device) 
model = MLP(NUM_INPUT).to(device) 

# Convert training data to torch tensor
train_X = torch.FloatTensor(X_train.to_numpy()).to(device) 
test_X = torch.FloatTensor(X_test.to_numpy()).to(device) 
y_train = torch.LongTensor(y_train.to_numpy()).squeeze().to(device) 

#print(y_train)

#y_test = torch.FloatTensor(y_test.values)
BATCH_SIZE = len(train_X)
datasets = torch.utils.data.TensorDataset(train_X, y_train)
train_iter = torch.utils.data.DataLoader(datasets, batch_size=BATCH_SIZE, shuffle=True)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LR)

for epoch in range(NUM_EPOCHS): 
  for x, y in train_iter:
    optimizer.zero_grad() 
    output = model(x) 
    #print(f'y_train is {y_train.shape}, output is {output.shape}')
    loss = criterion(output, y) 
    loss.backward() 
    optimizer.step() 
     
  if epoch % 100 == 0: 
      print(f"number of epoch {epoch}, loss = {loss.item()} ") 
      #net.zero_grad() 

predict_out = model(test_X)
_, y_pred = torch.max(predict_out, 1)
y_pred = [classes[y_pred.numpy()[i]] for i in range(len(y_pred))]
y_test = y_test.to_numpy().squeeze()
print('prediction accuracy =', accuracy_score(y_test, y_pred) )

#perm = PermutationImportance(model, random_state=1).fit(X_test, y_test)
display(show_weights(model))

#for i in range(len(y_test)):
#  print(f"test: {y_test[i]},\t pred: {predict_y[i]},\t result: {predict_y[i]==y_test[i]}")


mat = confusion_matrix(y_test, y_pred)
print('\n Confusion matrix ') 
print(mat)

df_y_pred = pd.DataFrame(y_pred)
#df_y_pred.to_csv('y_pred.csv',header=None,index=False)


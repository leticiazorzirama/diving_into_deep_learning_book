# Data preprocessing

# 2.2.1. Reading the Dataset

import os

# Get current path
path = os.getcwd()

# Make data directory and data file
os.makedirs(os.path.join(path, 'data'), exist_ok=True)
data_file = os.path.join(path, 'data', 'house_tiny.csv')
with open(data_file, 'w') as f:
    f.write('''NumRooms,RoofType,Price
NA,NA,127500
2,NA,106000
4,Slate,178100
NA,NA,140000''')

# Save data path
data_path = os.path.join(path, 'data/house_tiny.csv')
data_path

import pandas as pd

# Load data
data = pd.read_csv(data_path)
print(data)

# 2.2.2. Data Preparation
inputs, targets = data.iloc[:, 0:2], data.iloc[:, 2] # separate input form target

# handle missing values with imputation
inputs = pd.get_dummies(inputs, dummy_na = True) # get dummies for categorical values
print(inputs) 

inputs = inputs.fillna(inputs.mean()) # replace with the mean 
print(inputs)

2.2.2. Data Preparation

import torch
X = torch.tensor(inputs.to_numpy(dtype=float))
y = torch.tensor(targets.to_numpy(dtype=float))
X, y

# 2.2.5. Exercises
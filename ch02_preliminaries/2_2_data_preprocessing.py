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


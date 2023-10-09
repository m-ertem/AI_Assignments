#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 17:03:00 2023

@author: mustafa
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from sklearn.preprocessing import add_dummy_feature

from packaging import version
import sklearn

assert version.parse(sklearn.__version__) >= version.parse("1.0.1")

from pathlib import Path

IMAGES_PATH = Path() / "images" / "training_linear_models"
IMAGES_PATH.mkdir(parents=True, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = IMAGES_PATH / f"{fig_id}.{fig_extension}"
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)
    

# Replace 'imports-85.csv' with the actual file path if it's not in the current directory
dataset_url = "https://raw.githubusercontent.com/plotly/datasets/master/imports-85.csv"

# Read the CSV file into a Pandas DataFrame
df = pd.read_csv(dataset_url)

# Now, you can work with the 'df' DataFrame as needed
pd.set_option('display.max_columns', None)  # Show all columns
df.head(10)

target_variable_column = df[['city-mpg']] #target variable "y" column 
feature_columns = df[['curb-weight', 'engine-size']] #input x1 and x2 columns

# Convert selected columns to a NumPy array
y = target_variable_column.values # real mpg values
X = feature_columns.values # curb weights and engine sizes

X_b = add_dummy_feature(X)  # add x0 = 1 to each instance in order to have b term 
m = 204 # number of data we have
              
n_epochs = 50
lambda_hyper = 0.5 # hyperparameter to scale regularization term

np.random.seed(70) #default 42
theta = np.random.randn(3, 1)  # random initialization of weights and bias value

for epoch in range(n_epochs):
    for iteration in range(m):
        
        random_index = np.random.randint(m)
        xi = X_b[random_index : random_index + 1 ]
        yi = y[random_index : random_index + 1]
        #gradient calculation with L2 regularization
        gradients = 2 * xi.T / m @ (xi @ theta - yi) + lambda_hyper * theta
        #eta = learning_schedule(epoch * m + iteration)
        eta = 0.0000001
        theta = theta - eta * gradients

y_hat = X_b @ theta # predicted y values

plt.figure(figsize=(10, 4))  # formatting

# plots curb weight - mpg prediction
plt.subplot(121)
plt.plot(X[:,0], y_hat, "b.", color="blue") # prediction with blue dots
plt.plot(X[:,0], y, "b.", color="red")      # real values with red dots
plt.xlabel("$x1 (curb weights)$")
plt.ylabel("$y$", rotation=0)
plt.grid()

# plots engine size - mpg prediction
plt.subplot(122)
plt.plot(X[:,1], y_hat, "b.")  # prediction with blue dots
plt.plot(X[:,1], y, "b.", color="red") # real values with red dots

plt.xlabel("$x2 (engine sizes)$")
plt.ylabel("$y$", rotation=0)
plt.grid()
plt.show()

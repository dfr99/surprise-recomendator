from surprise import accuracy, Dataset, SVD
from surprise.reader import Reader
from surprise.model_selection import cross_validate, KFold, GridSearchCV, train_test_split
from surprise.trainset import Trainset

from surprise import (
    Dataset,    
    SVD,
    accuracy
)

import copy
from tabulate import tabulate

import numpy as np
import random

import pandas as pd

import matplotlib.pyplot as plt


file_path = '../data/ml-latest-small/ratings.csv'
df = pd.read_csv(file_path, delimiter=',')

user_column = 'userId'
product_column = 'movieId'
ratings_column = 'rating'

print('Número de usuarios:', len(df[user_column].value_counts()))
print('Número de productos:', len(df[product_column].value_counts()))
print('Número de puntuaciones:', len(df))

if df.isnull().any().any():
    print('Número de valores vacíos:\n', df.isnull().sum())
else:
    print('No existen valores vacíos (NA)')

if df.duplicated().any().any():
    print('Número de valores duplicados:', df.duplicated().sum())
else:
    print('No existen valores duplicados')

user_min_ratings = 20
product_min_ratings = 10

user_counts = df[user_column].value_counts()
product_counts = df[product_column].value_counts()

filtered_df = df[df[product_column].isin(user_counts[user_counts >= product_min_ratings].index)]

print('\nNúmero de productos después de filtrar los productos:', len(filtered_df[product_column].value_counts()))
num_rows, num_columns = filtered_df.shape
print('Número de filas:', num_rows)

filtered_df = filtered_df[filtered_df[product_column].isin(product_counts[product_counts >= product_min_ratings].index)]

print('Número de usuarios después de filtrar los usuarios:', len(filtered_df[user_column].value_counts()))
print('Número de productos despúes de filtrar los productos y los usuarios, respectivamente:', len(filtered_df[product_column].value_counts()))

num_rows, num_columns = filtered_df.shape
print('Número de filas:', num_rows)


plt.hist(filtered_df[user_column], bins=20, color='blue', edgecolor='black')

plt.title('Histogram of {}'.format(user_column))
plt.xlabel(user_column)
plt.ylabel('Frequency')

plt.show()

plt.hist(filtered_df[product_column], bins=20, color='blue', edgecolor='black')

plt.title('Histogram of {}'.format(product_column))
plt.xlabel(product_column)
plt.ylabel('Frequency')

plt.show()

plt.hist(filtered_df[user_column].mean(), bins=20, color='orange', edgecolor='black')

plt.title('Histogram of {}'.format(user_column))
plt.xlabel(user_column)
plt.ylabel('Frequency')

plt.show()

plt.hist(filtered_df[product_column].mean(), bins=20, color='orange', edgecolor='black')

plt.title('Histogram of {}'.format(product_column))
plt.xlabel(product_column)
plt.ylabel('Frequency')

plt.show()

value_counts = df[ratings_column].value_counts().sort_index()
value_counts.plot(kind='bar', color='green', edgecolor='black')

# Customize the plot
plt.title('Bar Diagram of {}'.format(ratings_column))
plt.xlabel(ratings_column)
plt.ylabel('Frequency')

# Show the plot
plt.show()
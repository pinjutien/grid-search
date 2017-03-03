import numpy as np
import pandas as pd
from preprocess import preprocessing_data
from model_params import classifier_param
from dklearn import DaskGridSearchCV
import sys
from time import time
from distributed import Executor

'''
This script is based on the following linked with customer modifiction to study grid serach in dask.
https://www.kaggle.com/svpons/airbnb-recruiting-new-user-bookings/script-0-8655/code

'''
#Loading data
paths = '/home/pinju/src/grid-search/airbnb-data/'
df_train = pd.read_csv(paths + 'train_users_2.csv')
df_test = pd.read_csv(paths + 'test_users.csv')

# preprocessing data
X, y, X_test = preprocessing_data(df_train, df_test)

#Classifier
# model_choice = 'DecisionTreeClassifier'
# model_choice = 'KNeighborsClassifier'
model_choice = 'random forests'
model = classifier_param[model_choice][0]
param_grid = classifier_param[model_choice][1]

print("model: ", model)
print("param_grid: ", param_grid)
# exc = Executor('173.208.222.74:8877', set_as_default=True)
for jobs in range(1):
    t0 = time()
    grid_search = DaskGridSearchCV(
        model,
        param_grid,
        # verbose= 3,
        # n_jobs = jobs,
        cv = 3)
    grid_search.fit(X, y)
    print(jobs, " ", time()-t0)

# grid_search.fit(X, y)
# grid_search.best_params_(X, y)

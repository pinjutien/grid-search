import numpy as np
import pandas as pd
from preprocess import preprocessing_data
from model_params import classifier_param
from sklearn.grid_search import GridSearchCV
import sys
from time import time

'''
This script is based on the following linked with customer modifiction to study grid serach in dask.
https://www.kaggle.com/svpons/airbnb-recruiting-new-user-bookings/script-0-8655/code

'''
#Loading data
paths = '~/grid-search/airbnb-data/'
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

for jobs in range(1,2):
    t0 = time()
    # print("start grid search: ", t0)
    grid_search = GridSearchCV(
        model,
        param_grid,
        verbose= 3,
        n_jobs = jobs,
        cv = 3
    )
    grid_search.fit(X, y)
    del grid_search
    # print("end grid search: ", time() -t0)
    print(jobs, " ", time()-t0)

# grid_search.fit(X, y)
# grid_search.best_params_(X, y)

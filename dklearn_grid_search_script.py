import numpy as np
import pandas as pd
from preprocess import preprocessing_data
from model_params import classifier_param
from dklearn import DaskGridSearchCV, DaskRandomizedSearchCV
import sys
from time import time
from distributed import Executor, Client
import dask

'''
This script is based on the following linked with customer modifiction to study grid serach in dask.
https://www.kaggle.com/svpons/airbnb-recruiting-new-user-bookings/script-0-8655/code

'''
#Loading data
paths = '/Users/pin-jutien/Documents/Working/grid-search/airbnb-data/'
df_train = pd.read_csv(paths + 'train_users_2.csv')
df_test = pd.read_csv(paths + 'test_users.csv')

# preprocessing data
X, y, X_test = preprocessing_data(df_train, df_test)

#Classifier
model_choice = 'DecisionTreeClassifier'
# model_choice = 'KNeighborsClassifier'
# model_choice = 'random forests'
model = classifier_param[model_choice][0]
param_grid = classifier_param[model_choice][1]

print("model: ", model)
print("param_grid: ", param_grid)
scheduler_address = '173.208.222.74:8877'
runing_time = []
n_workers = 1
print("number of workers: ", n_workers)

for cv_temp in range(2,5):
    t0 = time()
    if(model_choice == 'random forests'):
        print("\n\n")
        print("\t\t Special Grid Search for random foreset")
        print("\n\n")
        c = Client(scheduler_address, set_as_default=True)
        grid_search = DaskRandomizedSearchCV(
            model,
            param_grid,
            cv = cv_temp,
            get= c.get
        )
    else:
        c = Client(scheduler_address, set_as_default=True)
        grid_search = DaskGridSearchCV(
            model,
            param_grid,
            cv = cv_temp
        )

    grid_search.fit(X, y)
    time_elapse = time()-t0
    runing_time.append([model_choice, cv_temp, time_elapse])
    print(" running time: ", time_elapse)
    num_graph = len(grid_search.dask_graph_)
    print(" size of graph: ", num_graph)

    
    runing_time_df = pd.DataFrame(data = runing_time, columns = ['model', 'cv', 'time'])
    runing_time_df['n_workers'] = n_workers
    runing_time_df['n_graph'] = num_graph
    runing_time_df.to_csv("output/dk_learn_runningtime_workers_{n_workers}_cv_{cv_temp}.csv".format(n_workers = n_workers, cv_temp = cv_temp))
    del grid_search, runing_time_df
# grid_search.fit(X, y)
# grid_search.best_params_(X, y)

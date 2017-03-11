#!/usr/bin/python
# -*- coding: utf-8 -*-
# vim: tabstop=4 shiftwidth=4 softtabstop=4 colorcolumn=80 expandtab

__author__       = 'Zhuo Yin'
__copyright__    = 'Copyright 2017'

import pandas as pd
from xgboost.sklearn import *
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

def main(argc, argv):
    '''
    grid = {
        'max_depth'         : range(1,12,2),
        'learning_rate'     : [0.01, 0.03, 0.05, 0.07, 0.09, 0.1, 0.2],
        'n_estimators'      : [100, 300, 400, 500, 600, 700, 800, 1000],
        'gamma'             : [0.5, 1, 1.5],
        #'min_child_weight'
        #'max_delta_step'
        'subsample'         : [0.2,0.4,0.66,0.8,1],
        'colsample_bytree'  : [0.25,0.5,0.7,0.9,1],
        #'colsample_bylevel': [0.2, 0.5, 0.7, 0.9, 1]
        'reg_alpha'         : np.arange(0.01, 1, 0.05)
        #'reg_lambda'       : np.arange(0.01, 1, 0.05)
        #'scale_pos_weight'
        #'base_score'
    },
    '''

    grid = {
        'max_depth'         : range(1,12),
        'min_samples_split' : range(2,10),
        'min_samples_leaf' : range(2,10)
    }

    
    # model = XGBRegressor()
    model = DecisionTreeRegressor()
    store = pd.HDFStore('foobar.hdf5', 'r')

    train_x = store['train_x']
    train_y = store['train_y']

    print('train_x shape: {}'.format(train_x.shape))
    print('train_y shape: {}'.format(train_y.shape))

    from distributed import Executor, Client
    from dklearn import DaskGridSearchCV

    print('doing grid search')
    scheduler_address = '173.208.222.74:8877'
    # exc = Executor(scheduler_address, set_as_default=True)
    c = Client(scheduler_address, set_as_default=True)    
    gs = DaskGridSearchCV(model, grid).fit(train_x, train_y)
    # c = Client(scheduler_address, set_as_default=True)
    #gs = DaskGridSearchCV(model, grid, get = c.get).fit(train_x, train_y)


    pass

if __name__ == '__main__':
    import sys
    main(len(sys.argv), sys.argv)

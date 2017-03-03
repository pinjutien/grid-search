import numpy as np
from xgboost.sklearn import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

classifier_param = {
    'xgboost' : [
        XGBClassifier(),
        {
            'max_depth'        : [ 6, 7],
            'learning_rate'    : [0.3],
            'n_estimators'     : [25],
            'subsample'        :[0.5],
            'colsample_bytree' :[0.5]
        }
    ],

    'random forests': [
        RandomForestClassifier(),
        {
            'max_depth'    : np.arange(1, 5),
            'n_estimators' :np.arange(5, 20),
            'max_features' : ['auto', 'sqrt', 'log2', None],
        }
    ],

    'DecisionTreeClassifier': [
        DecisionTreeClassifier(),
    {
        'max_depth'        : np.arange(1, 5),
        'random_state'     : np.arange(1, 20),
        'max_features'     : ['auto', 'sqrt', 'log2', None],
        'min_samples_leaf' : np.arange(1, 5)
    }
    ],

    'AdaBoostClassifier': [
        AdaBoostClassifier(),
        {
            'n_estimators' :np.arange(20, 60),
            'learning_rate'    : np.arange(0.2, 1.0, 0.1),
        }
    ],

    'KNeighborsClassifier': [
        KNeighborsClassifier(),
        {
            # memory issue
            'n_neighbors' : np.arange(2, 9),
            'weights' : ['uniform', 'distance'],
            'algorithm'    : ['auto', 'ball_tree', 'kd_tree', 'brute'],
            'leaf_size'    : np.arange(10, 40),

            # 'weights' : ['uniform'],
            # 'algorithm'    : ['auto'],
            # 'leaf_size'    : np.arange(10, 40),
        }
    ],
}

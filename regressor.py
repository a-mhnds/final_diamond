import numpy as np
import pandas as pd
import os
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedGroupKFold
from sklearn.metrics import confusion_matrix, r2_score, mean_squared_error, mean_absolute_error
from typing import Literal
import json
import yaml
import tqdm

from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
from livelossplot import PlotLossesKeras
from keras.callbacks import  CSVLogger
import matplotlib.pyplot as plt

param_grid_svr = {'C':np.linspace(0.8,1.2,4), 'kernel':['linear', 'poly', 'rbf', 'sigmoid'], 'gamma':['scale', 'auto']}
param_grid_knn = {'n_neighbors':range(2,12)}
param_grid_decisionTree = {'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson']}
param_grid_randomForest = {'n_estimators':range(100,200,50), 'criterion':['squared_error', 'absolute_error', 'friedman_mse', 'poisson']}
# param_grid_logreg = {'max_iter':range(100,1000,100), 'tol':np.linspace(1e-5, 1e-3, 10), 'solver': ['lbfgs', 'newton-cg', 'newton-cholesky', 'sag', 'saga']}


class Regressor:
    def __init__(self):
        self.model_name = [KNeighborsRegressor(),
                DecisionTreeRegressor(),
                SVR(),
                RandomForestRegressor()]
        
        self.param_grid_lst = [param_grid_knn,
                  param_grid_decisionTree,
                  param_grid_svr,
                  param_grid_randomForest]


    def train_test_val(self, X, y, TrainSize, ValSize):
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=TrainSize, random_state=42)
        X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=ValSize, random_state=42)
        self.X_train = X_train
        self.X_test = X_test
        self.X_val = X_val
        self.y_train = y_train
        self.y_test = y_test
        self.y_val = y_val
 


    def estimator(self, score_metric: Literal['r2', 'mse', 'mae', 'rmse'], *ann_args):

        os.makedirs('out_files', exist_ok=True)

        if not ann_args:
            ann_args = [10, 8,"adam", 'relu', 32, 10] #activation function should change - number of nodes should change.

        model = Sequential()
        i = 0
        while i<ann_args[0]:
            model.add(Dense(units=ann_args[1], activation=ann_args[3]))
            i += 1

        model.add(Dense(1)) # activation function should be changed to the one for regression.
        model.compile(optimizer=ann_args[2], loss='mean_squared_error', metrics=['mse', 'mae', tf.keras.metrics.R2Score()])
        
        csvlogger = CSVLogger(filename='out_files/reg_cnn_metrics.csv', append=True)
        plotlosses = PlotLossesKeras()
        history = model.fit(self.X_train, self.y_train, batch_size=ann_args[4], epochs=ann_args[5], validation_data=(self.X_val, self.y_val),
                            callbacks=[plotlosses, csvlogger], verbose=0)
        plt.savefig('out_files/reg_ann_metrics_plot.jpg')
        y_predict = model.predict(self.X_test)
        if score_metric=='r2':
            train_score = history.history['r2_score'][-1]
            val_score = history.history['_'.join(['val','r2_score'])][-1]
        elif score_metric=='mse':
            train_score = history.history['mean_squared_error'][-1]
            val_score = history.history['_'.join(['val','mean_squared_error'])][-1]
        elif score_metric=='rmse':
            train_score = np.sqrt(history.history['mean_squared_error'][-1])
            val_score = np.sqrt(history.history['_'.join(['val','mean_squared_error'])][-1])
        elif score_metric=='mae':
            train_score = history.history['mean_absolute_error'][-1]
            val_score = history.history['_'.join(['val','mean_absolute_error'])][-1]
        score = [train_score, val_score]
        print('ANN scores: ', score)


        print('model list:', self.model_name)
        model_rows = []
        for model in tqdm.tqdm(self.model_name):
            print('model: ', model)
            grid=GridSearchCV(model, self.param_grid_lst[self.model_name.index(model)], verbose=0, refit = True)
            grid.fit(self.X_val, self.y_val)
            best_parameters = grid.best_params_
            print("Best parameters found:", best_parameters)

            model_class = type(model)

            json_file_path = "".join(['out_files/', model_class.__name__,'.json'])
            with open(json_file_path,'w') as f:
                json.dump(best_parameters, f, indent=4)

            model = grid.best_estimator_
            model.fit(self.X_train, self.y_train)
            y_predict = model.predict(self.X_test)
            if score_metric =='r2':
                score = r2_score(self.y_test, y_predict)
            elif score_metric == 'mse':
                score = mean_squared_error(self.y_test, y_predict)
            elif score_metric =='mae':
                score = mean_absolute_error(self.y_test, y_predict)
            elif score_metric == 'rmse':
                score = np.sqrt(mean_squared_error(self.y_test, y_predict))

            row_data = {'model_name':model_class.__name__, ''.join(['score(',score_metric,')']):score}
            model_rows.append(row_data)
            print('model', score_metric, 'score: ', score)
        model_results_df = pd.DataFrame(model_rows)
        model_results_df.to_csv('out_files/reg_model_scores.csv', index=False)
        return y_predict, score
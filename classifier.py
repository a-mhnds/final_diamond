'''
This scipt examines a set of classification models to determine the model with the best score.
Hyperparameters are grid searched.
Tables and plots of scores are stored in the out_files folder.

Author: Ali Mohandesi
Date: 30-07-2025
'''


import numpy as np
import pandas as pd
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedGroupKFold
from sklearn.metrics import confusion_matrix
import json
import yaml
import tqdm

from keras.models import Sequential
from keras.layers import Dense
from livelossplot import PlotLossesKeras
from keras.callbacks import  CSVLogger
import matplotlib.pyplot as plt

# Lists of parameters for the GridSearch CV method
param_grid_svc = {'C':np.linspace(0.8,1.2,4), 'kernel':['linear', 'poly', 'rbf', 'sigmoid'], 'gamma':['scale', 'auto']}
param_grid_knn = {'n_neighbors':range(2,12)}
param_grid_decisionTree = {'criterion':['gini', 'entropy', 'log_loss']}
param_grid_randomForest = {'n_estimators':range(100,200,50), 'criterion':['gini', 'entropy', 'log_loss']}
# param_grid_logreg = {'max_iter':range(100,1000,100), 'tol':np.linspace(1e-5, 1e-3, 10), 'solver': ['lbfgs', 'newton-cg', 'newton-cholesky', 'sag', 'saga']}


class Classifier:
    def __init__(self):
        self.model_name = [KNeighborsClassifier(),
                DecisionTreeClassifier(),
                RandomForestClassifier(),
                SVC()]
        
        self.param_grid_lst = [param_grid_knn,
                  param_grid_decisionTree,
                  param_grid_randomForest,
                  param_grid_svc]


    # split the dataframe into train, test, and validation
    def train_test_val(self, X, y, TrainSize, ValSize):
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=TrainSize, random_state=42)
        X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=ValSize, random_state=42)
        self.X_train = X_train
        self.X_test = X_test
        self.X_val = X_val
        self.y_train = y_train
        self.y_test = y_test
        self.y_val = y_val
    


    # estimate the best classifier model 
    def estimator(self,*ann_args):

        # create the out_files directory for storing generated figures and tables
        os.makedirs('out_files', exist_ok=True)

        # default ANN hyperparameters
        if not ann_args:
            ann_args = [10, 8,"adam", 'relu', 32, 10]

        model = Sequential()
        i = 0
        # add hidden layers
        while i<ann_args[0]:
            model.add(Dense(units=ann_args[1], activation=ann_args[3]))
            i += 1

        # add output layer
        model.add(Dense(units=5, activation='softmax'))

        model.compile(optimizer=ann_args[2], loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
        # create a log file from the loss and score at each epoch
        csvlogger = CSVLogger(filename='out_files/cls_cnn_metrics.csv', append=True)

        # create loss and score plot 
        plotlosses = PlotLossesKeras()

        history = model.fit(self.X_train, self.y_train, batch_size=ann_args[4], epochs=ann_args[5], validation_data=(self.X_val, self.y_val),
                            callbacks=[plotlosses, csvlogger], verbose=0)
        
        plt.savefig('out_files/cls_ann_metrics_plot.jpg')
        y_predict = model.predict(self.X_test)
        train_acc = history.history['accuracy'][-1]
        val_acc = history.history['val_accuracy'][-1]
        score = [train_acc, val_acc]
        print('ANN scores: ', score)


        # try all models to find the one with the best score
        print('model list:', self.model_name)
        model_rows = []
        for model in tqdm.tqdm(self.model_name):
            print('model: ', model)

            # grid search for the best set of hyperparameters for each model
            grid=GridSearchCV(model, self.param_grid_lst[self.model_name.index(model)], verbose=0, refit = True)
            grid.fit(self.X_val, self.y_val)
            best_parameters = grid.best_params_
            print("Best parameters found:", best_parameters)

            model_class = type(model)
            
            # save best hyperparameters for the model
            json_file_path = "".join(['out_files/', model_class.__name__, '_params', '.json'])
            with open(json_file_path,'w') as f:
                json.dump(best_parameters, f, indent=4)
            
            model = grid.best_estimator_
            model.fit(self.X_train, self.y_train)
            y_predict = model.predict(self.X_test)
            score = model.score(self.X_test,self.y_test)
            row_data = {'model_name':model_class.__name__, 'score(accuracy)':score}
            model_rows.append(row_data)
            print('model score: ', score)

            # create and save confusion matrix for the model.
            cm = confusion_matrix(self.y_test, y_predict)
            print('Cnfusion Matrix for ', model_class.__name__, "is:\n", cm)
            cm_txt_file_path = "".join(['out_files/', model_class.__name__, '_cm', '.txt'])
            with open(cm_txt_file_path, 'w') as f:
                f.write(str(cm))

        # store list of models with their scores in a data from and save it in a CSV file
        model_results_df = pd.DataFrame(model_rows)
        model_results_df.to_csv('out_files/cls_model_scores.csv', index=False)
        return y_predict, score
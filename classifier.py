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


    def train_test_val(self, X, y, TrainSize, ValSize):
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=TrainSize, random_state=42)
        X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=ValSize, random_state=42)
        self.X_train = X_train
        self.X_test = X_test
        self.X_val = X_val
        self.y_train = y_train
        self.y_test = y_test
        self.y_val = y_val
 
    # def log_reg_estimator(self):
    #     model = LogisticRegression()
    #     model.fit(X_train, y_train)
    #     y_predict = model.predict(X_test)
    #     return y_predict, model.score(X_test,y_test)
    

    # def knn_classifier_estimator(self, X_train, X_test, y_train, y_test, k):
    #     model = KNeighborsClassifier(n_neighbors=k)
    #     model.fit(X_train, y_train)
    #     y_predict = model.predict(X_test)
    #     return y_predict, model.score(X_test,y_test)
    

    # def decisionTree_classifier_estimator(self, X_train, X_test, y_train, y_test, method):
    #     model = DecisionTreeClassifier(criterion=method)
    #     model.fit(X_train, y_train)
    #     y_predict = model.predict(X_test)
    #     return y_predict, model.score(X_test,y_test)
    

    # def randomForest_classifier_estimator(self, X_train, X_test, y_train, y_test, method, estimators):
    #     model = RandomForestClassifier(n_estimators=estimators, criterion=method)
    #     model.fit(X_train, y_train)
    #     y_predict = model.predict(X_test)
    #     return y_predict, model.score(X_test,y_test)
    

    # def svc_classifier_estimator(self):
    #     grid=GridSearchCV(SVC(), param_grid_svc, verbose=0, refit = True)
    #     grid.fit(self.X_val, self.y_val)
    #     model = grid.best_estimator_
    #     model.fit(self.X_train, self.y_train)
    #     y_predict = model.predict(self.X_test)
    #     return y_predict, model.score(self.X_test,self.y_test)
    

    # def ann_classifier_estimator(self, X_train, X_test, y_train, y_test, n_layer, n_unit, opt, act_func, batch, epoch):

    #     model = Sequential()
    #     while n_layer:
    #         model.add(Dense(units=n_unit, activation=act_func))
    #     model.add(Dense(units=5, activation='softmax'))
    #     model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    #     history = model.fit(X_train, y_train, batch_size=batch, epochs=epoch)

    #     y_predict = model.predict(X_test)
    #     train_acc = history.history['accuray'][-1]
    #     val_acc = history.history['val_accuray'][-1]

    #     return y_predict, train_acc, val_acc
    


    def estimator(self,*ann_args):

        os.makedirs('out_files', exist_ok=True)

        if not ann_args:
            ann_args = [10, 8,"adam", 'relu', 32, 10]

        model = Sequential()
        i = 0
        while i<ann_args[0]:
            model.add(Dense(units=ann_args[1], activation=ann_args[3]))
            i += 1

        model.add(Dense(units=5, activation='softmax'))
        model.compile(optimizer=ann_args[2], loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
        csvlogger = CSVLogger(filename='out_files/cls_cnn_metrics.csv', append=True)
        plotlosses = PlotLossesKeras()
        history = model.fit(self.X_train, self.y_train, batch_size=ann_args[4], epochs=ann_args[5], validation_data=(self.X_val, self.y_val),
                            callbacks=[plotlosses, csvlogger], verbose=0)
        plt.savefig('out_files/cls_ann_metrics_plot.jpg')
        y_predict = model.predict(self.X_test)
        train_acc = history.history['accuracy'][-1]
        val_acc = history.history['val_accuracy'][-1]
        score = [train_acc, val_acc]
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
            score = model.score(self.X_test,self.y_test)
            row_data = {'model_name':model_class.__name__, 'score(accuracy)':score}
            model_rows.append(row_data)
            print('model score: ', score)
        model_results_df = pd.DataFrame(model_rows)
        model_results_df.to_csv('out_files/cls_model_scores.csv', index=False)
        return y_predict, score
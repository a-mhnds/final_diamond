'''
This script examines a set of clusterring models to determine the model with the best silhouette score.
Hyperparameters are searched for each model.
Tables of models are stored in the out_files folder.

Author: Ali Mohandesi
Date: 30-07-2025
'''


import tqdm
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, MeanShift, AgglomerativeClustering
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import silhouette_score

# list of hyperparameters for each clusterring model
param_kmean = {'n_clusters':range(2,30)}
param_meanshift = {'bandwidth':np.linspace(0.5,10,10)}
param_hier = {'n_clusters':range(2,30)}
class Cluster:
    def __init__(self):
        pass

    
    # split the dataframe into train, and test
    def train_test(self, X, TrainSize):
        X_train, X_test = train_test_split(X, train_size=TrainSize, random_state=42)
        self.X_train = X_train
        self.X_test = X_test
    

    
    def create_cluster(self):

        model_rows = []
        best_score = -100
        print('model: MeanShift')
        # determine the best bandwidth for the method mean shift
        for bw in tqdm.tqdm(param_meanshift['bandwidth']):
            model = MeanShift(bandwidth=bw)
            labels = model.fit_predict(self.X_test)
            if len(np.unique(labels))==1:
                continue
            else:
                score = silhouette_score(self.X_test, labels)
                if score > best_score:
                    best_score = score
                    best_bw = bw
        print('MeanShift bandWidth: ', best_bw)

        # train the mean shift model using the bandwidth found in the previous step
        model = MeanShift(bandwidth=best_bw)
        labels = model.fit_predict(self.X_train)
        score = silhouette_score(self.X_train, labels)
        print('MeanShift score: ', score)
        row_data = {'model_name':'MeanShift', 'score':score, 'labels':len(np.unique(labels))}
        model_rows.append(row_data)
        print(np.unique(labels))
        print('len of lebels', len(labels))


        best_score = -100
        # score = -100
        print('model: AgglomerativeClustering')
        # determine the number of clusters for the agglomerative clusterring method
        for nc in tqdm.tqdm(param_hier['n_clusters']):
            model = AgglomerativeClustering(n_clusters=nc)
            labels = model.fit_predict(self.X_test)
            score = silhouette_score(self.X_test, labels)
            if score > best_score:
                best_score = score
                best_nc = nc
       
        print('AgglomerativeClustering n_clusters: ', best_nc)
        # train the agglomerative clusterring model using the number of clusters found in the previous step
        model = AgglomerativeClustering(n_clusters=best_nc)
        labels = model.fit_predict(self.X_train)
        score = silhouette_score(self.X_train, labels)
        print('AgglomerativeClustering score: ', score)
        row_data = {'model_name':'AgglomerativeClustering', 'score':score, 'labels':len(np.unique(labels))}
        model_rows.append(row_data)
        print(np.unique(labels))



        best_score = -100
        # score = -100
        print('model: KMeans')
        # determine the number of clusters for the k-means clusterring method
        for nc in tqdm.tqdm(param_kmean['n_clusters']):
            model = KMeans(n_clusters=nc)
            labels = model.fit_predict(self.X_test)
            score = silhouette_score(self.X_test, labels)
            if score > best_score:
                best_score = score
                best_nc = nc
      
        print('KMeans n_clusters: ', best_nc)

        # train the k-means clusterring model using the number of clusters found in the previous step
        model = KMeans(n_clusters=best_nc)
        labels = model.fit_predict(self.X_train)
        score = silhouette_score(self.X_train, labels)
        print('KMeans score: ', score)
        row_data = {'model_name':'KMeans', 'score':score, 'labels':len(np.unique(labels))}
        model_rows.append(row_data)
        print(np.unique(labels))

        # store the clusterring models, their scores, and number of labels in a dataframe and save it in a CSV file in the out_files folder
        model_results_df = pd.DataFrame(model_rows)
        model_results_df.to_csv('out_files/cluster_model_scores.csv', index=False)
import tqdm
import numpy as np
from sklearn.cluster import KMeans, MeanShift, AgglomerativeClustering
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import silhouette_score


param_kmean = {'n_clusters':range(2,10)}
param_meanshift = {'bandwidth':np.linspace(0.5,5,2)}
param_hier = {'n_clusters':range(2,10)}
class Cluster:
    def __init__(self):
        pass

    
    def train_test(self, X, TrainSize):
        X_train, X_test = train_test_split(X, train_size=TrainSize, random_state=42)
        self.X_train = X_train
        self.X_test = X_test
    

    
    def create_cluster(self):

        best_score = -100
        print('model: MeanShift')
        for bw in tqdm.tqdm(param_meanshift['bandwidth']):
            model = MeanShift(bandwidth=bw)
            labels = model.fit_predict(self.X_test)
            score = silhouette_score(self.X_test, labels)
            if score > best_score:
                best_score = score
                best_bw = bw
        print('MeanShift bandWidth: ', best_bw)
        model = MeanShift(bandwidth=best_bw)
        labels = model.fit_predict(self.X_train)
        print('MeanShift score: ', silhouette_score(self.X_train, labels))


        best_score = -100
        print('model: AgglomerativeClustering')
        for nc in tqdm.tqdm(param_hier['n_clusters']):
            model = AgglomerativeClustering(n_clusters=nc)
            labels = model.fit_predict(self.X_test)
            score = silhouette_score(self.X_test, labels)
            if score > best_score:
                best_score = score
                best_nc = nc
        print('AgglomerativeClustering n_clusters: ', best_nc)
        model = AgglomerativeClustering(n_clusters=best_nc)
        labels = model.fit_predict(self.X_train)
        print('AgglomerativeClustering score: ', silhouette_score(self.X_train, labels))



        best_score = -100
        print('model: KMeans')
        for nc in tqdm.tqdm(param_kmean['n_clusters']):
            model = KMeans(n_clusters=nc)
            labels = model.fit_predict(self.X_test)
            score = silhouette_score(self.X_test, labels)
            if score > best_score:
                best_score = score
                best_nc = nc
        print('KMeans n_clusters: ', best_nc)
        model = KMeans(n_clusters=best_nc)
        labels = model.fit_predict(self.X_train)
        print('KMeans score: ', model.inertia_)
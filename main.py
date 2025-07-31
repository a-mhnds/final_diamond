import os
# to suppresses INFO messages from tensorflow
os.environ['TF_CPP_MIN_LoG_LEVEL']='1'

# Calling Analyzer class and its methods
from analyzer import Analyzer
analyzer = Analyzer()
df = analyzer.read_dataset("data/diamonds.csv")
df = analyzer.shuffle(df)
df = analyzer.volume(df)


# Classification
# refined_df, target = analyzer.refine_dataframe(df,['price'])
# from classifier import Classifier
# cls = Classifier()
# cls.train_test_val(refined_df,target, 0.8, 0.5)
# y_predict, score = cls.estimator()


# Regression
# from regressor import Regressor
# reg = Regressor()
# reg.train_test_val(refined_df,target, 0.8, 0.5)
# y_predict, score = reg.estimator(score_metric='r2')


# Clusterring
# refined_df = analyzer.refine_dataframe(df)
# from clustering import Cluster
# cls = Cluster()
# cls.train_test(refined_df, 0.8)
# cls.create_cluster()
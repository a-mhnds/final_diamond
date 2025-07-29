import os
os.environ['TF_CPP_MIN_LoG_LEVEL']='1'

from analyzer import Analyzer
analyzer = Analyzer()
df = analyzer.read_dataset("data/diamonds.csv")

df = analyzer.shuffle(df)
# target = df.cut.to_numpy()
# df.drop('cut', axis=1, inplace=True)

refined_df, target = analyzer.refine_dataframe(df,['price'])

# analyzer0.plot_correlationMatrix(df)
# analyzer0.plot_histograms_categorical(df, 'clarity')
# analyzer0.plot_boxPlot(df, 'clarity')
# analyzer0.plot_pairPlot(df, 'cut')


# from classifier import Classifier
# cls = Classifier()
# cls.train_test_val(refined_df,target, 0.8, 0.5)
# y_predict, score = cls.estimator()


# from clustering import Cluster
# cls = Cluster()
# cls.train_test(refined_df, 0.8)
# cls.create_cluster()

from regressor import Regressor
reg = Regressor()
reg.train_test_val(refined_df,target, 0.8, 0.5)
y_predict, score = reg.estimator(score_metric='r2')
import pandas as pd
class Analyzer:
    def __init__(self):
        pass


    def read_dataset(self, csv_file):

        df = pd.read_csv(csv_file, index_col=0)
        # print(f"Five rows of the dataframe created from the given CSV file:\n{df.head(5)}")
        # print(f"Basic statistics of the dataframe:\n{df.describe()}")
        return df


    def drop_missing_data(self, df):
        df = df.dropna()
        print(f"Dataframe after dropping missing values:\n{df.head(5)}")
        return df
    

    def drop_columns(self, df, columns):
        df.drop(columns=columns, inplace=True)
        print(f"Dataframe after dropping specified columns:\n{df.head(5)}")
        return df
    

    def shuffle(self, df):
        df = df.sample(frac=1,random_state=42)
        df = df.reset_index(drop=True)
        return df

    
    def refine_dataframe(self, df, target=None):
        from sklearn.preprocessing import StandardScaler
        from sklearn.preprocessing import LabelEncoder
        import numpy as np
        
        if target:
            target = df[target]
            print('df:', df.head(5))
            df.drop(target, axis=1, inplace=True)
            target = LabelEncoder().fit_transform(target.values.ravel())

        df_num = df.select_dtypes(exclude='object')
        scaler = StandardScaler()
        df_num = scaler.fit_transform(df_num)

        df_cat = df.select_dtypes(include='object')
        df_cat = pd.get_dummies(df_cat)

        refined_df = np.concat([df_num, df_cat], axis=1)

        if not target.any():
            return refined_df
        else:
            return refined_df, target
    

    def plot_correlationMatrix(self, df):
        import matplotlib.pyplot as plt
        import seaborn as sns

        df_num = df.select_dtypes(exclude='object')
        corr_mat = df_num.corr()

        fig, ax = plt.subplots(figsize=(11, 9))
        # cmap = sns.diverging_palette(230, 20, as_cmap=True)
        sns.heatmap(corr_mat, center=0,
            square=True, linewidths=.5, annot=True , cbar_kws={"shrink": .5, "label":"Corr. Coef."})
        fig.savefig('out_files/corr_matrix.jpg')
        plt.show()


    def plot_histograms_categorical(self, df, feature):
        import matplotlib.pyplot as plt
        import seaborn as sns

        # df_cat = df.select_dtypes(include='object')
        
        fig, ax = plt.subplots(figsize=(11, 9))
        sns.histplot(data=df, x=feature, stat='percent', color='k')
        str = ['Histogram of', feature]
        str = " ".join(str)
        str_fig_name = 'out_files/' + 'Histogram_' + feature + '.jpg'
        ax.set_title(str)
        fig.savefig(str_fig_name)
        plt.show()


    def plot_boxPlot(self, df, feature1, feature2=None):
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        fig, ax = plt.subplots(figsize=(11, 9))

        sns.boxplot(data=df, x=feature1, y=feature2, color='k')
        str_fig_name = 'out_files/' + 'boxplot_' + feature1 + '.jpg'
        str = ['Boxplot of', feature1]
        str = " ".join(str)
        ax.set_title(str)
        fig.savefig(str_fig_name)
        plt.show()


    def plot_pairPlot(self, df, feature=None):
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        fig, ax = plt.subplots()
        sns.pairplot(data=df, hue=feature)
        if feature!=None:
            str_fig_name = 'out_files/' + 'paiplot_' + feature + '.jpg'
            str = ['Pair plot of', feature]
        else:
            str_fig_name = 'out_files/' + 'paiplot_' + '.jpg'
            str = 'Pair plot'
        str = " ".join(str)
        ax.set_title(str)
        fig.savefig(str_fig_name)
        plt.show()
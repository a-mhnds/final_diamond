"""
This script is for pre-process a dataset from a CSV file to be used in machine learning methods or visualization of
the data using various plotting techniques.
Author: Ali Mohandesi
Date: 30-07-2025
"""


import pandas as pd
class Analyzer:
    def __init__(self):
        pass


    # Read dataset from a CSV file
    def read_dataset(self, csv_file):
        df = pd.read_csv(csv_file, index_col=0)
        return df


    # Remove rows with missing data
    def drop_missing_data(self, df):
        df = df.dropna()
        print(f"Dataframe after dropping missing values:\n{df.head(5)}")
        return df
    

    # Remove a given feature from the data frame
    def drop_columns(self, df, columns):
        df.drop(columns=columns, inplace=True)
        print(f"Dataframe after dropping specified columns:\n{df.head(5)}")
        return df
    

    # Shuffle rows of the data frame
    def shuffle(self, df):
        df = df.sample(frac=1,random_state=42)
        df = df.reset_index(drop=True)
        return df
    
    # Replace x, y, and z  features with volume
    def volume(self, df):
        df['volume'] = df['x']*df['y']*df['z']
        df.drop(['x','y','z'], axis=1, inplace= True)
        return df


    
    # Refine the data frame: Standardize numerical and encode nominal features
    def refine_dataframe(self, df, target=None):
        from sklearn.preprocessing import StandardScaler
        from sklearn.preprocessing import LabelEncoder
        import numpy as np
        
        
        
        if target == ['price']:
            target = df[target]
            # print('df:', df.head(5))
            df.drop(target, axis=1, inplace=True)
            target = target.values.ravel()
        elif target:
            target = df[target]
            # print('df:', df.head(5))
            df.drop(target, axis=1, inplace=True)
            target = LabelEncoder().fit_transform(target.values.ravel())

        df_num = df.select_dtypes(exclude='object')
        scaler = StandardScaler()
        df_num = scaler.fit_transform(df_num)

        df_cat = df.select_dtypes(include='object')
        df_cat = pd.get_dummies(df_cat)

        refined_df = np.concat([df_num, df_cat], axis=1)

        if target is None:
            return refined_df
        else:
            return refined_df, target
    

    # plot the correlation matrix
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


    # Plot histogram for a categorical feature
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


    # Plot box plot
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


    # Plot a pair plot
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
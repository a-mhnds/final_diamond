import pandas as pd
class Analyzer:
    def __init__(self):
        pass


    def read_dataset(self, csv_file):

        df = pd.read_csv(csv_file, index_col=0)
        print(f"Five rows of the dataframe created from the given CSV file:\n{df.head(5)}")
        print(f"Basic statistics of the dataframe:\n{df.describe()}")
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

    
    def refine_dataframe(self, df, target):
        from sklearn.preprocessing import StandardScaler
        from sklearn.preprocessing import LabelEncoder
        import numpy as np
        
        target = df[target]
        df.drop(target, axis=1, inplace=True)

        df_num = df.select_dtypes(exclude='object')
        scaler = StandardScaler()
        df_num = scaler.fit_transform(df_num)

        df_cat = df.select_dtypes(include='object')
        df_cat = pd.get_dummies(df_cat)

        target = LabelEncoder().fit_transform(target)
        refined_df = np.concat([df_num, df_cat], axis=1)

        return refined_df, target
    
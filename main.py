from analyzer import Analyzer
analyzer0 = Analyzer()
df = analyzer0.read_dataset("data/diamonds.csv")

analyzer0.refine_dataframe(df,['clarity'])

analyzer0.get_name

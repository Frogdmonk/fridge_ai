import pandas as pd

df = pd.read_csv("recipe.csv")
print(df.columns.tolist())
print(df.shape)
print(df.head(2))

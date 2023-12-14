import pandas as pd

df = pd.read_csv('final_entry_cat.csv', encoding='cp949', dtype={'Code': str})
df = df.iloc[:, :7]
df.to_csv("final_entry_cat_7cols.csv", encoding='cp949')
print('here')
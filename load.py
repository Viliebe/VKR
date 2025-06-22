import vaex
import pandas as pd

df = vaex.from_csv('processed_data.csv', sep=',', convert=True, low_memory=False)  # Для точки с запятойa
# df = vaex.open('processed_data.hdf5')
print(df)


#открыть хдф5:
#file_path = 'panel.hdf5'
file_path = 'results/result.hdf5'
df = vaex.open(file_path)
print(df)

# file_path = 'processed_encoded.hdf5'
file_path = 'processed_data.hdf5'
#file_path = 'results/result.hdf5'
df = vaex.open(file_path)
df_pd = df.head(100).to_pandas_df()
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 0)
print(df_pd)

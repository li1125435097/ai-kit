import pandas as pd
import numpy as np

# 读取csv文件
rdf = pd.read_csv("titanic.csv")
num = rdf.shape[0]
rdf['Tel'] = np.arange(num)
print(rdf)

rdf.rename(columns={'Name': 'Name1'}, inplace=True)
print(rdf)
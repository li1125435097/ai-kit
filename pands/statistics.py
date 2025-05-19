import pandas as pd
import numpy as np

# 读取csv文件
rdf = pd.read_csv("titanic.csv")

# 分类统计
# td = rdf[["Age","Sex"]].groupby("Sex").mean()
# td = rdf.groupby("Sex")['Age'].mean()
td = rdf.groupby("Sex").mean(numeric_only=True)

print(td)

# 分类统计2
td = rdf["Age"].value_counts()
print(td)
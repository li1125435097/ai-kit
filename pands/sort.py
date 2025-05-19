import pandas as pd
import numpy as np

# 读取csv文件py
rdf = pd.read_csv("titanic.csv")

print(rdf.sort_values(by="Age", ascending=False))
print(rdf.sort_values(by=["Age", "Sex"]))
print(rdf.sort_values(by=["Age", "Sex"], ascending=[False, True]))
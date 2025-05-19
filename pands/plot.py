import pandas as pd
import matplotlib.pyplot as plt

# 读取csv文件
rdf = pd.read_csv("titanic.csv")
ages = rdf['Age']


rdf.plot()
rdf.plot.bar()
rdf.plot.hist()
plt.show()
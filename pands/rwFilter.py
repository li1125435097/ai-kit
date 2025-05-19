import pandas as pd

# pands库主要操作两个对象，一个是Series，一个是DataFrame
# df = pd.DataFrame(
#     {
#         "Name": [
#             "Braund, Mr. Owen Harris",
#             'Allen, "Mr. William Henry',
#             "Bonnell, Miss. Elizabeth",
#         ],
#         "Age": [22, 35, 58],
#         "Sex": ["male", "male", "female"],
#     }
# )

# 写入csv文件
# df.to_csv("titanic.csv", index=False)

# 写入html文件
# rdf.to_html("titanic.html")
# rdf.to_json("titanic.json")

# 读取csv文件
rdf = pd.read_csv("titanic.csv")
ages = rdf['Age']
# print(ages.describe())
print(rdf[rdf['Age'] > 50])
# print(rdf[rdf['Age'].isin([22, 58])])
# print(rdf[rdf['Age'].between(22, 50)])

# 50岁以上的名字
print(rdf.loc[rdf['Age']>50,'Name'])

# 选择单元格，行，列
print(rdf.iloc[1:2, 1:2])
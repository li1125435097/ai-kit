import pandas as pd

# pands库主要操作两个对象，一个是Series，一个是DataFrame
df = pd.DataFrame(
    {
        "Name": [
            "Braund, Mr. Owen Harris",
            "Allen, Mr. William Henry",
            "Bonnell, Miss. Elizabeth",
        ],
        "Age": [22, 35, 58],
        "Sex": ["male", "male", "female"],
    }
)


print(df.shape)
print(df.head(2))
print(df.describe())
print(df.info())
print(df.columns)
print(df.values)
print(df.index)
print(df.dtypes)
print(df.ndim)
print(df.size)
print(df.axes)
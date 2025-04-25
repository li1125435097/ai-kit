import sklearn as sk

# 设置assume_finite为True，即假设数据是有限的
with sk.config_context(assume_finite=True):
    # 打印当前的sklearn配置
    print(sk.get_config())

# 打印当前的sklearn配置
print(sk.get_config())

# 设置working_memory为256
sk.set_config(working_memory=256)
print('')
# 打印当前的sklearn配置的键
print(sk.get_config().keys())

# 打印sklearn的版本信息
sk.show_versions()

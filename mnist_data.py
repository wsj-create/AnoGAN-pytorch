#导入相关包
import numpy as np
import pandas as pd




"""
mnist数据集读取
"""
## 读取训练集数据  （60000，785）
train = pd.read_csv(".\data\mnist_train.csv",dtype = np.float32)
## 读取测试集数据  （10000，785）
test = pd.read_csv(".\data\mnist_test.csv",dtype = np.float32)


# 查询训练数据中标签为7、8的数据，并取前400个
train = train.query("label in [7.0, 8.0]").head(400)

# 查询训练数据中标签为7、8的数据，并取前400个
test = test.query("label in [2.0, 7.0, 8.0]").head(600)

# 取除标签后的784列数据
train = train.iloc[:,1:].values.astype('float32')
test = test.iloc[:,1:].values.astype('float32')

# train:(400,784)-->(400,28,28)
# test:(600,784)-->(600,28,28)
train = train.reshape(train.shape[0], 28, 28)
test = test.reshape(test.shape[0], 28, 28)



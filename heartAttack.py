from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pandas as pd

import tensorflow as tf

from tensorflow import feature_column
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

URL = 'https://storage.googleapis.com/applied-dl/heart.csv'
dataframe = pd.read_csv(URL)
dataframe.head()

train, test = train_test_split(dataframe, test_size=0.2)
train, val = train_test_split(train, test_size=0.2)
print(len(train), 'train examples')
print(len(val), 'validation examples')
print(len(test), 'test examples')


# 一种从Pandas Dataframe创建tf.data数据集的使用方法
def df_to_dataset(dataframe, shuffle=True, batch_size=32):
  dataframe = dataframe.copy()
  labels = dataframe.pop('target')
  ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
  if shuffle:
    ds = ds.shuffle(buffer_size=len(dataframe))
  ds = ds.batch(batch_size)
  return ds

batch_size = 5 # 小批量用于演示目的
train_ds = df_to_dataset(train, batch_size=batch_size)
val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)

for feature_batch, label_batch in train_ds.take(1):
  print('Every feature:', list(feature_batch.keys()))
  print('A batch of ages:', feature_batch['age'])
  print('A batch of targets:', label_batch )

example_batch = next(iter(train_ds))[0]

# 用于创建特征列和转换批量数据
def demo(feature_column):
  feature_layer = layers.DenseFeatures(feature_column)
  print(feature_layer(example_batch).numpy())

age = feature_column.numeric_column("age")
demo(age)

#One hot 形式 将年龄分为几类
age_buckets = feature_column.bucketized_column(age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])
demo(age_buckets)

#thal表示为字符串（例如“固定”，“正常”或“可逆”），我们无法直接将字符串提供给模型，相反，我们必须首先将它们映射到数值。分类词汇表列提供了一种将字符串表示为独热矢量的方法
thal = feature_column.categorical_column_with_vocabulary_list(
      'thal', ['fixed', 'normal', 'reversible'])

thal_one_hot = feature_column.indicator_column(thal)
demo(thal_one_hot)

# 请注意，嵌入列的输入是我们先前创建的分类列
#假设我们不是只有几个可能的字符串，而是每个类别有数千（或更多）值。由于多种原因，随着类别数量的增加，使用独热编码训练神经网络变得不可行，我们可以使用嵌入列来克服此限制。
#嵌入列不是将数据表示为多维度的独热矢量，而是将数据表示为低维密集向量，其中每个单元格可以包含任意数字，而不仅仅是0或1.嵌入的大小（在下面的例子中是8）是必须调整的参数。
thal_embedding = feature_column.embedding_column(thal, dimension=8)
demo(thal_embedding)
from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt


fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

train_images, test_images = train_images/255.0, test_images/255.0

train_images = train_images[..., tf.newaxis]
test_images = test_images[..., tf.newaxis]

#使用tf.data批处理和随机打乱数据集：
# train_ds = tf.data.Dataset.from_tensor_slices(
#     (train_images, train_labels)).shuffle(10000).batch(32)
# test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(32)


model = keras.Sequential([

    keras.layers.Conv2D(32, 3, activation='relu'),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')

])
optimizer = tf.keras.optimizers.Adam(lr=0.001)

reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.1, patience=10, verbose=0, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)

# monitor：被监测的量
# factor：每次减少学习率的因子，学习率将以lr = lr*factor的形式被减少
# patience：当patience个epoch过去而模型性能不提升时，学习率减少的动作会被触发
# mode：‘auto’，‘min’，‘max’之一，在min模式下，如果检测值触发学习率减少。在max模式下，当检测值不再上升则触发学习率减少。
# epsilon：阈值，用来确定是否进入检测值的“平原区”
# cooldown：学习率减少后，会经过cooldown个epoch才重新进行正常操作
# min_lr：学习率的下限


model.compile(
optimizer = optimizer,
loss = 'sparse_categorical_crossentropy',
metrics = ['accuracy']
)
model.fit(train_images, train_labels, epochs=1, callbacks=[reduce_lr])

test_loss, test_acc = model.evaluate(test_images, test_labels)
print('\nTest accuracy:', test_acc)

from __future__ import absolute_import, division, print_function, unicode_literals

import os

import tensorflow as tf
from tensorflow import keras

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

train_labels = train_labels[:1000]
test_labels = test_labels[:1000]

print(train_images[0])

train_images = train_images[:1000].reshape(-1, 28 * 28) / 255.0
test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.0

print(train_images[0])

def create_model():
  model = tf.keras.models.Sequential([
    keras.layers.Dense(512, activation='relu', input_shape=(784,)),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10, activation='softmax')
  ])

  model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

  return model


# 创建基本模型实例
# model = create_model()
# model.summary()

checkpoint_path = "Model/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# 创建一个检查点回调
# cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
#                                                  save_weights_only=True,
#                                                  verbose=1)
#
# model = create_model()
#
# model.fit(train_images, train_labels,  epochs = 10,
#           validation_data = (test_images,test_labels))
#
# model.save('Model/my_model.h5')

# model = create_model()
# loss, acc = model.evaluate(test_images, test_labels)
# print("Restored model, accuracy: {:5.2f}%".format(100*acc))
#
# model.load_weights(checkpoint_path)
# loss,acc = model.evaluate(test_images, test_labels)
# print("Restored model, accuracy: {:5.2f}%".format(100*acc))

# new_model = keras.models.load_model('Model/my_model.h5')
# loss, acc = new_model.evaluate(test_images, test_labels)
# print("Restored model, accuracy: {:5.2f}%".format(100*acc))

import time
saved_model_path = "./Model/{}".format(int(time.time()))

model = create_model()

model.fit(train_images, train_labels, epochs=5)

tf.keras.experimental.export_saved_model(model, saved_model_path)
# saved_model_path

new_model = tf.keras.experimental.load_from_saved_model(saved_model_path)
new_model.summary()

# 必须要在评估之前编译模型
# 如果仅部署已保存的模型，则不需要此步骤

new_model.compile(optimizer=model.optimizer,  # keep the optimizer that was loaded
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 评估加载后的模型
loss, acc = new_model.evaluate(test_images, test_labels)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))
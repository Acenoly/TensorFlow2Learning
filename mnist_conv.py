import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Helper libraries
import numpy as np
import tensorflow as tf
from tensorflow import keras

# train_images = mnist.train.images

# for label in mnist.train.labels:
#     print(label)
    # train_labels= mnist.train.labels
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train / 255.0
x_test = x_test / 255.0

x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

print(x_train[0].shape)

model = keras.Sequential([
    keras.layers.Conv2D(32, 3,activation='relu', input_shape=(28, 28, 1)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, 3, activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, 3, activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=1)



test_loss, test_acc = model.evaluate(x_test, y_test)
print('\nTest accuracy:', test_acc)
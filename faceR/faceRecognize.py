import pathlib

import tensorflow as tf
from tensorflow import keras

data_root = pathlib.Path("G:/facedatahandle")

for item in data_root.iterdir():
  print(item)

all_image_paths = list(data_root.glob('*/*'))
all_image_paths = [str(path) for path in all_image_paths]

image_count = len(all_image_paths)
print(all_image_paths[:10])

label_names = sorted(item.name for item in data_root.glob('*/') if item.is_dir())
print(label_names)

label_to_index = dict((name, index) for index, name in enumerate(label_names))

all_image_labels = [label_to_index[pathlib.Path(path).parent.name]
                    for path in all_image_paths]

def preprocess_image(image):
  image = tf.image.decode_jpeg(image, channels=3)
  image = tf.image.resize(image, [160, 160])
  image /= 255.0  # normalize to [0,1] range

  return image

def load_and_preprocess_image(path):
  image = tf.io.read_file(path)
  return preprocess_image(image)

import matplotlib.pyplot as plt

image_path = all_image_paths[0]
label = all_image_labels[0]


img_path = "G:/facedatahandle/000/cut_000_0.jpg"
img_raw = tf.io.read_file(img_path)
img_tensor = tf.image.decode_image(img_raw)

print(img_tensor.shape)
print(img_tensor.dtype)

import matplotlib.pyplot as plt

image_path = all_image_paths[0]
label = all_image_labels[0]


path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
image_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)

label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(all_image_labels, tf.int64))
image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))

BATCH_SIZE = 32

# 设置一个和数据集大小一致的 shuffle buffer size（随机缓冲区大小）以保证数据
# 被充分打乱。
ds = image_label_ds.shuffle(buffer_size=image_count)
ds = ds.repeat()
ds = ds.batch(BATCH_SIZE)
# 当模型在训练的时候，`prefetch` 使数据集在后台取得 batch。
ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

print(ds)

model = keras.Sequential([
    keras.layers.Conv2D(32, 11, activation='relu', input_shape=(160, 160, 3)),
    keras.layers.Conv2D(32, 74, activation='relu'),
    keras.layers.Conv2D(16, 9, activation='relu'),
    keras.layers.Conv2D(16, 9, activation='relu'),
    keras.layers.Conv2D(16, 31, activation='relu'),
    keras.layers.Conv2D(16, 5, activation='relu'),
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dense(500, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(ds, epochs=50, steps_per_epoch=50)


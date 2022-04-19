# package importing

# update package
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import pathlib
from numbers_extraction import id_numbers_locate

data_dir = pathlib.Path('../trainer')

batch_size = 32
img_height = 180
img_width = 180

train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

# Each class represent a category
print('the categories are: ', train_ds.class_names)

for image_batch, labels_batch in train_ds:
    print(image_batch.shape)
    print(labels_batch.shape)
    break

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

normalization_layer = layers.Rescaling(1. / 255)

normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
# Notice the pixel values are now in `[0,1]`.
print(np.min(first_image), np.max(first_image))

num_classes = len(train_ds.class_names)

model = Sequential([
    layers.Rescaling(1. / 255, input_shape=(img_height, img_width, 3)),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses
              .SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# model.summary()


# numbers of times the AI is trained
epochs = 100

# This is where we train the model
# history = model.fit(
#     train_ds,
#     validation_data=val_ds,
#     epochs=epochs
# )

data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal",
                          input_shape=(img_height,
                                       img_width,
                                       3)),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ]
)

model = Sequential([
    data_augmentation,
    layers.Rescaling(1. / 255),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses
              .SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


def card_classification(path):
    """
    detects what type of ID the given ID is;
    Biometric, non-biometric, driving license, (passport too)
    But in order to make it work, we need a bigger dataset.

    Args:
        path(str): the path to the image which contains the ID


    Returns:
    A set of positions in an array, by which the program can locate
    the ID number. For example, in the biometric ID,
    """
    # Image.open(path).show()
    img = tf.keras.utils.load_img(
        path, target_size=(img_height, img_width)
    )
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    predictions = model.predict(img_array)
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    print(
        "This image most likely belongs to {} {:.2f} % confidence.".format
        (train_ds.class_names[np.argmax(score)], 100 * np.max(score))
    )

    # coordinates must be changed accordingly
    # coordinates = [StartY, EndY, StartX, EndX]
    # Note that StartY starts from the bottom of the image
    # Note that StartX starts from the left of the image
    #
    # Note that when we multiply the position with a number larger than 1
    # The cuts a part of the image, making it smaller
    # Meanwhile, when we multiply witha number smaller than one
    # we take more part of the original image
    # https://i.ibb.co/T1GG5NT/Coordinates-Image.png
    # View this link to visualize
    #
    #


    if 100 * np.max(score) > 0:
        coordinates = []
        print(train_ds.class_names[np.argmax(score)], 'with',
              100 * np.max(score), 'confidence')
        if train_ds.class_names[np.argmax(score)] == 'Biometric_ID':
            coordinates = [1, 1.5, 0.9, 1.1]
        if train_ds.class_names[np.argmax(score)] == 'Driving_License':
            coordinates = [1, 1.5, 0.9, 1.1]
        if train_ds.class_names[np.argmax(score)] == 'Non_Biometric_ID':
            coordinates = [1, 1.5, 0.9, 1.1]
        if train_ds.class_names[np.argmax(score)] == 'Passports':
            coordinates = [1, 1.5, 0.9, 1.1]
        if train_ds.class_names[np.argmax(score)] == 'Visot':
            coordinates = [1, 1.5, 0.9, 1.1]
        return id_numbers_locate(path, coordinates)
    else:
        return 'please take another pic'

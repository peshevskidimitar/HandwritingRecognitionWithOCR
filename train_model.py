import argparse

import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# print(tf.__version__)

if __name__ == '__main__':
    # Parse the command line arguments.
    parser = argparse.ArgumentParser(description='Train deep neural network for optical character recognition (OCR).')
    parser.add_argument(dest='file_path', metavar='file-path', type=str, help='The file path to the whole dataset.')
    args = parser.parse_args()

    # Load the dataset as Pandas data frame.
    dataset = pd.read_csv(args.file_path, header=None)

    # Extract the labels and images from the dataset.
    dataset_labels = dataset.values[:, 0]
    dataset_images = dataset.values[:, 1:]

    # Reshape the images with 28 x 28 dimensions.
    dataset_images = np.reshape(dataset_images, (dataset_images.shape[0], 28, 28, 1))

    # Split the data into train and test sets in the ratio of 80:20.
    train_images, test_images, train_labels, test_labels = train_test_split(dataset_images, dataset_labels,
                                                                            test_size=0.20)

    # Define the deep neural network model.
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(len(np.unique(train_labels)), activation='softmax')
    ])

    # Compile the above-defined model.
    model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=1e-4),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    print(model.summary())

    # Create two image generators for data augmentation (one for the train set, and one for the test dataset).
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.2,
        horizontal_flip=False,
        fill_mode='nearest'
    )
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow(train_images, train_labels, batch_size=50, shuffle=True)
    validation_generator = test_datagen.flow(test_images, test_labels, batch_size=50, shuffle=True)

    # Fit the deep neural network.
    model.fit(
        train_generator,
        steps_per_epoch=500,
        epochs=100,
        validation_data=validation_generator,
        validation_steps=50,
        verbose=2)

    # Save the trained model.
    model.save('optical-character-recognizer')

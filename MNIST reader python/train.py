import cv2
import numpy as np
import tensorflow as tf
import keras


def train_new(layer1: int, layer2: int, layer3: int, epochs: int):
    mnist = keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Images: 28x28 are resized to 14x14
    # Color value is 0-255, but we will use 0-1

    x_train = np.array([cv2.resize(image, (14, 14)) for image in x_train])
    x_test = np.array([cv2.resize(image, (14, 14)) for image in x_test])

    x_train = keras.utils.normalize(x_train, axis = 1)
    x_test = keras.utils.normalize(x_test, axis = 1)

    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(32, (3, 3), activation = 'relu', input_shape = (14, 14, 1)))
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Conv2D(64, (3, 3), activation = 'relu'))
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Conv2D(64, (3, 3), activation = 'relu'))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64, activation = 'relu'))
    model.add(keras.layers.Dense(10, activation = 'softmax'))

    model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

    model.fit(x_train, y_train, epochs = epochs)

    model.save(f"../models/MNIST/mnist_handwritten_{layer1}x{layer2}x{layer3}_{epochs}_.keras")


def train(path: str, epochs: int) -> keras.models.Model:
    mnist = keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Images: 28x28 are resized to 14x14
    # Color value is 0-255, but we will use 0-1

    x_train = np.array([cv2.resize(image, (14, 14)) for image in x_train])
    x_test = np.array([cv2.resize(image, (14, 14)) for image in x_test])

    x_train = keras.utils.normalize(x_train, axis = 1)
    x_test = keras.utils.normalize(x_test, axis = 1)

    model: keras.models.Model = keras.models.load_model(path)

    model.fit(x_train, y_train, epochs = epochs)

    return model


def train_save_as(path: str, epochs: int, path2: str):
    mnist = keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Images: 28x28 are resized to 14x14
    # Color value is 0-255, but we will use 0-1

    x_train = np.array([cv2.resize(image, (14, 14)) for image in x_train])
    x_test = np.array([cv2.resize(image, (14, 14)) for image in x_test])

    x_train = keras.utils.normalize(x_train, axis = 1)
    x_test = keras.utils.normalize(x_test, axis = 1)

    model: keras.models.Model = keras.models.load_model(path)

    model.fit(x_train, y_train, epochs = epochs)

    model.save(path2)

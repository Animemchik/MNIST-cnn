import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras

from train import train, train_new, train_save_as

# train_new(16, 16, 1)
# train_new(16, 16, 10)
# train_new(16, 16, 100)
#
# train_new(128, 128, 1)
# train_new(128, 128, 10)
# train_new(128, 128, 100)
#
# train_new(1024, 512, 10)
#
# train_new(32, 24, 16, 100)
#
# train_new(64, 64, 64, 20)

mnist = keras.datasets.mnist
(_, _), (x_test, y_test) = mnist.load_data()

x_test_original = x_test
x_test = np.array([cv2.resize(image, (14, 14)).reshape(1, 14, 14, 1) for image in x_test])

model: keras.models.Model = keras.models.load_model("../models/MNIST/mnist_handwritten_CNN_60_epochs.keras")
model.summary()


def custom_tests():
    image_number = 1
    while os.path.isfile(f"test_handwrites/digit_{image_number}.png"):
        try:
            image = cv2.imread(f"test_handwrites/digit_{image_number}.png", cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, (14, 14)).reshape(1, 14, 14, 1)
            image = np.invert(np.array([image]))
            image = image / 255.0
            prediction = model.predict(image)
            plt.imshow(image[0], cmap=plt.cm.binary)
            plt.title(f"The digit №{image_number} is probably {np.argmax(prediction)}")
            mng = plt.get_current_fig_manager()
            mng.window.state('zoomed')
            plt.show()
        except Exception as e:
            print(e)
        finally:
            image_number += 1


def test_1000_images():
    correct = 0
    wrong = 0
    for i in range(1000):
        if np.argmax(model.predict(x_test[i], verbose = 0)) == y_test[i]:
            correct += 1
        else:
            wrong += 1
        if (i + 1) % 10 == 0:
            print(i + 1)
        # if (i + 1) % 50 == 0:
        #     plt.imshow(x_test_original[i], cmap = plt.cm.binary)
        #     plt.title(
        #         (f"The digit №{i + 1} is probably {np.argmax(model.predict(x_test[i].reshape(1, 14, 14, 1), verbose = 0))}."
        #          f"The correct value is {y_test[i]}")
        #     )
        #     mng = plt.get_current_fig_manager()
        #     mng.window.state('zoomed')
        #     plt.show()

    print(f"Correct: {correct} Wrong: {wrong}")


def test_2000_images():
    correct = 0
    wrong = 0
    for i in range(2000):
        if np.argmax(model.predict(x_test[i], verbose = 0)) == y_test[i]:
            correct += 1
        else:
            wrong += 1
        if (i + 1) % 100 == 0:
            print(i + 1)

    print(f"Correct: {correct} Wrong: {wrong}")


def random_80_tests():
    random_indices = np.random.choice(len(x_test), size=80, replace=False)

    fig, axes = plt.subplots(8, 10, figsize=(15, 12))
    fig.suptitle("Random MNIST Predictions", fontsize=20)

    for ax, idx in zip(axes.flat, random_indices):
        image = x_test[idx]
        true_label = y_test[idx]
        prediction = np.argmax(model.predict(image.reshape(1, 14, 14, 1), verbose=0))

        image = x_test_original[idx]
        if prediction != true_label:
            image = np.stack([image] * 3, axis = -1)
            image[..., 1:] = 0

        ax.imshow(image, cmap=plt.cm.binary)
        if prediction == true_label:
            ax.set_title(f"Prediction: {prediction}\nTrue: {true_label}")
        else:
            ax.set_title(f"Prediction: {prediction}\nTrue: {true_label}", color='red')
        ax.axis("off")

    # Показ графика
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()


# custom_tests()
# test_2000_images()
# test_1000_images()
random_80_tests()

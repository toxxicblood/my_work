import cv2
import numpy as np
import os
import sys
import tensorflow as tf

from sklearn.model_selection import train_test_split

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # Get a compiled neural network
    model = get_model()

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    model.evaluate(x_test, y_test, verbose=2)

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")


def load_data(data_dir):
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.

    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` should
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.
    """
    images = []
    labels = []
    for category in range(NUM_CATEGORIES - 1):

        category_path = os.path.join(data_dir, str(category))

        if not os.path.isdir(category_path):  # skip unreadable dirs
            continue
        for filename in os.listdir(category_path):
            img_path = os.path.join(category_path, filename)
            img = cv2.imread(img_path)  # reads in bgr format

            if img is None:  # skip unreadable files
                continue

            # resize image
            img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
            # print(img.shape)
            # convert to rgb
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # store img and corresponding label
            images.append(img / 255.0)
            labels.append(category)
    # images = np.array(images)
    # print(images)
    # print(labels)
    return (images, labels)


def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """
    model = tf.keras.models.Sequential()
    # the input shape of the images
    model.add(tf.keras.Input(shape=(IMG_WIDTH, IMG_HEIGHT, 3)))
    # here im gonna experiment with different activation functions, filters
    # and strides..
    model.add(tf.keras.layers.Conv2D(128, 5, strides=2, activation="elu"))
    model.add(tf.keras.layers.MaxPool2D(pool_size=3))
    # model.add(tf.keras.layers.Conv2D(128, 3, activation="relu"))
    # model.add(tf.keras.layers.MaxPool2D(pool_size=2))
    model.summary()

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(300, activation="elu"))
    model.add(tf.keras.layers.Dropout(0.5))

    # add an output layer with units for all digits
    model.add(
        tf.keras.layers.Dense(
            NUM_CATEGORIES - 1,
            activation="softmax",
        )
    )

    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    model.summary()

    return model


if __name__ == "__main__":
    main()

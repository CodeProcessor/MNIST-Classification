'''
Created on 7/21/20

@author: dulanj
'''
import numpy as np
from matplotlib import pyplot as plt
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.utils import to_categorical


class LoadData():
    def __init__(self):
        (train, self.train_labels), (test, self.test_labels) = self.load()

        train_norm = train.astype('float32')
        test_norm = test.astype('float32')
        # normalize to range 0-1
        self.train_images = train_norm / 255.0
        self.test_images = test_norm / 255.0

        print(self.train_images.shape)
        print(self.train_labels.shape)
        print(self.test_images.shape)
        print(self.test_labels.shape)
        # self.display(self.train_images[0])

    def load_raw(self):
        mnist_data_set = tf.keras.datasets.mnist
        (x_train, trainY), (x_test, testY) = mnist_data_set.load_data()
        x_train = x_train.astype('float32') / 255.
        x_test = x_test.astype('float32') / 255.
        x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
        x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
        print(x_train.shape)
        print(x_test.shape)
        return (x_train, trainY), (x_test, testY)

    def load(self):
        mnist_data_set = tf.keras.datasets.mnist
        (trainX, trainY), (testX, testY) = mnist_data_set.load_data()
        # Reshape data
        trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
        testX = testX.reshape((testX.shape[0], 28, 28, 1))
        # one hot encode target values
        trainY = to_categorical(trainY)
        testY = to_categorical(testY)
        return (trainX, trainY), (testX, testY)

    def display(self, pixels):
        plt.imshow(pixels, cmap='gray')
        plt.show()


if __name__ == "__main__":
    obj = LoadData()
    obj.load()
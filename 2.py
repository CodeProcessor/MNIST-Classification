'''
Created on 7/18/20

@author: dulanj
'''
import numpy as np
from tensorflow import keras
from loaddata import LoadData
import tensorflow as tf
from sklearn.model_selection import KFold
from model import load_model

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)


class Classifier(LoadData):

    def __init__(self):
        super(Classifier, self).__init__()
        self.model = None
    
    def add_noise(self, x_train, x_test):
        noise_factor = 0.5
        x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
        x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)
        x_train_noisy = np.clip(x_train_noisy, 0, 1)
        x_test_noisy = np.clip(x_test_noisy, 0, 1)
        return x_train_noisy, x_test_noisy
    
    def train(self):
        n_folds = 5
        kfold = KFold(n_folds, shuffle=True, random_state=1)
        self.train_images, self.test_images = self.add_noise(self.train_images, self.test_images)
        for k, (train_ix, test_ix) in enumerate(kfold.split(self.train_images)):
            # define and load model
            model = load_model()
            # select rows for train and test
            trainX, trainY, testX, testY = self.train_images[train_ix], self.train_labels[train_ix], \
                                           self.train_images[test_ix], self.train_labels[test_ix]
            # fit model
            history = model.fit(trainX, trainY, epochs=5, batch_size=32, validation_data=(testX, testY), verbose=1)
            # evaluate model
            _, acc = model.evaluate(self.test_images, self.test_labels, verbose=0)
            print('Fold {} Accuracy> {}'.format(k+1, acc * 100.0))
            self.model = model

    def test_accuracy(self):
        pred_y = self.model.predict(self.test_images)


    def main(self):
        self.train()


if __name__ == "__main__":
    obj = Classifier()
    obj.main()
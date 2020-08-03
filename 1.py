'''
Created on 7/18/20

@author: dulanj
'''
import tensorflow as tf
from sklearn.model_selection import KFold
from tensorflow.keras.utils import plot_model

from loaddata import LoadData
from model import load_model

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)


class Classifier(LoadData):

    def __init__(self):
        super(Classifier, self).__init__()
        self.model = None

    def train_with_k_fold(self):
        n_folds = 5
        kfold = KFold(n_folds, shuffle=True, random_state=1)
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
            print('Fold {} Accuracy> {}'.format(k + 1, acc * 100.0))
            self.model = model

    def train_normal(self):
        self.model = load_model()
        self.save_plotted_model()
        each_epoch = dict()
        fp = open('accuracy.csv', 'w')
        fp.write('epoch, loss,accuracy,val_loss, val_accuracy\n')
        for epoch in range(25):
            history = self.model.fit(self.train_images, self.train_labels, epochs=1, batch_size=32,
                                     validation_data=(self.test_images, self.test_labels), verbose=1)
            msg = '{},'.format(epoch)
            for _key in history.history.keys():
                if _key not in each_epoch:
                    each_epoch[_key] = []
                each_epoch[_key].append(history.history[_key][0])
                msg += '{:.4f},'.format(history.history[_key][0])

            fp.write(msg[:-1] + '\n')
        fp.close()
        print(each_epoch)

    def save_plotted_model(self):

        print("Saving plotted model")
        plot_model(self.model, to_file='model.png', show_shapes=True)

    def test_accuracy(self):
        pred_y = self.model.predict(self.test_images)

    def main(self):
        self.train_normal()


if __name__ == "__main__":
    obj = Classifier()
    obj.main()

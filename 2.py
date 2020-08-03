'''
Created on 7/18/20

@author: dulanj
'''
import tensorflow as tf
from sklearn.model_selection import KFold

from loaddata import LoadData
from model import add_noise
from model import load_model

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)


class Classifier(LoadData):

    def __init__(self):
        super(Classifier, self).__init__()
        self.model = None

    def train_normal(self):
        """:arg
        Different noise values tested against the model training
        """
        fp = open('accuracy_2.csv', 'w')
        fp.write('noise, loss,accuracy,val_loss, val_accuracy\n')
        for nf in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.5, 2]:
            train_images, test_images = add_noise(self.train_images, self.test_images, noise_factor=nf)
            model = load_model(0)
            epoch_range = 5
            print("Noise Factor: {}".format(nf))
            for epoch in range(epoch_range):
                history = model.fit(train_images, self.train_labels, epochs=1, batch_size=32,
                                    validation_data=(test_images, self.test_labels), verbose=0)
                msg = '{},'.format(nf)
                if epoch == epoch_range - 1:
                    for _key in history.history.keys():
                        msg += '{:.4f},'.format(history.history[_key][0])
                    # Write results back to a file
                    fp.write(msg[:-1] + '\n')
                    print(msg)
        fp.close()


if __name__ == "__main__":
    obj = Classifier()
    obj.train_normal()

'''
Created on 7/18/20

@author: dulanj
'''
import tensorflow as tf

from loaddata import LoadData
from model import add_noise
from model import load_model, create_encoder_decorder_model_v2

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)


class Classifier(LoadData):

    def __init__(self):
        super(Classifier, self).__init__()
        self.model = None
        self.auto_encoder_model = None

    def train_encorder_decoder_network_v2(self):
        self.auto_encoder_model = create_encoder_decorder_model_v2()
        x_train_noisy, x_test_noisy = add_noise(self.train_images, self.test_images, 1)
        self.auto_encoder_model.fit(x_train_noisy, self.train_images,
                                    epochs=50,
                                    batch_size=256,
                                    shuffle=True,
                                    validation_data=(x_test_noisy, self.test_images))

    def train_with_denoised_data(self):
        """:arg
        Using autoencoders denoise the data
        Then train using the CNN model
        """
        self.model = load_model()

        x_train_noisy, x_test_noisy = add_noise(self.train_images, self.test_images, noise_factor=1)

        x_train_denoised = self.auto_encoder_model.predict(x_train_noisy)
        x_test_denoised = self.auto_encoder_model.predict(x_test_noisy)

        for epoch in range(25):
            history = self.model.fit(x_train_denoised, self.train_labels, epochs=1, batch_size=32,
                                     validation_data=(x_test_denoised, self.test_labels), verbose=1)

    def train_with_noised_data(self):
        """:arg

        Without denoising directly train using the noisy data"""
        self.model = load_model()

        x_train_noisy, x_test_noisy = add_noise(self.train_images, self.test_images, noise_factor=1)

        for epoch in range(25):
            history = self.model.fit(x_train_noisy, self.train_labels, epochs=1, batch_size=32,
                                     validation_data=(x_test_noisy, self.test_labels), verbose=1)


if __name__ == "__main__":
    obj = Classifier()
    obj.train_encorder_decoder_network_v2()
    obj.train_with_denoised_data()

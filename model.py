'''
Created on 8/3/20

@author: dulanj
'''
import os

import numpy as np
import tensorflow as tf


def load_model(verbose=1):
    inputs = tf.keras.Input(batch_size=32, shape=(28, 28, 1))
    x = tf.keras.layers.Conv2D(24, (5, 5), activation=tf.nn.relu)(inputs)
    x = tf.keras.layers.MaxPool2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(16, (3, 3), activation=tf.nn.relu)(x)
    x = tf.keras.layers.MaxPool2D((2, 2))(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(64, activation=tf.nn.relu)(x)
    x = tf.keras.layers.Dense(16, activation=tf.nn.relu)(x)
    outputs = tf.keras.layers.Dense(10, activation=tf.nn.sigmoid)(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    # compile model
    opt = tf.keras.optimizers.SGD(lr=0.01, momentum=0.9, decay=0.0001)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    if verbose == 1:
        print(model.summary())

    return model


def create_encoder_decorder_model():
    # this is the size of our encoded representations
    encoding_dim = 32  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats

    # this is our input placeholder
    input_img = tf.keras.layers.Input(shape=(784,))
    # "encoded" is the encoded representation of the input
    encoded = tf.keras.layers.Dense(encoding_dim, activation='relu')(input_img)
    # "decoded" is the lossy reconstruction of the input
    decoded = tf.keras.layers.Dense(784, activation='sigmoid')(encoded)

    # this model maps an input to its reconstruction
    autoencoder = tf.keras.Model(input_img, decoded)
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
    return autoencoder


def create_encoder_decorder_model_v2():
    # Input
    x = tf.keras.layers.Input(name='inputs', shape=(28, 28, 1), dtype='float32')
    o = x

    # Encoder
    o = tf.keras.layers.Conv2D(32, (3, 3), activation=tf.keras.layers.LeakyReLU(alpha=0.1), padding='same',
                               strides=(1, 1), name='en_conv1')(o)
    o = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='en_pool1')(o)
    o = tf.keras.layers.Conv2D(32, (3, 3), activation=tf.keras.layers.LeakyReLU(alpha=0.1), padding='same',
                               strides=(1, 1), name='en_conv2')(o)
    enc = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='en_pool2')(o)

    # Decoder
    o = tf.keras.layers.Conv2D(32, (3, 3), activation=tf.keras.layers.LeakyReLU(alpha=0.1), padding='same',
                               strides=(1, 1), name='de_conv1')(enc)
    o = tf.keras.layers.UpSampling2D((2, 2), name='upsampling1')(o)
    o = tf.keras.layers.Conv2D(32, (3, 3), activation=tf.keras.layers.LeakyReLU(alpha=0.1), padding='same',
                               strides=(1, 1), name='de_conv2')(o)
    o = tf.keras.layers.UpSampling2D((2, 2), name='upsampling2')(o)
    dec = tf.keras.layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same', strides=(1, 1), name='de_conv3')(o)

    # Print network summary
    tf.keras.Model(inputs=x, outputs=dec).summary()

    model = tf.keras.Model(inputs=x, outputs=dec)
    model.compile(optimizer='adadelta', loss='binary_crossentropy')

    weights_name = 'autoenc3_weight.bin'
    # model.load_weights('autoenc3_weight.bin')
    if os.path.exists(weights_name):
        model.load_weights(weights_name)

    return model


def add_noise(x_train, x_test, noise_factor=0.5):
    x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
    x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)
    x_train_noisy = np.clip(x_train_noisy, 0, 1)
    x_test_noisy = np.clip(x_test_noisy, 0, 1)
    return x_train_noisy, x_test_noisy

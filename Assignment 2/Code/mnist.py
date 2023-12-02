# %% [markdown]
# **NN Classification**
# 
# Gian Favero | ECSE 556 | December 1st, 2023

# %%
# Import MNIST dataset
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Preprocess the data
train_images = train_images.reshape((60000, 28*28)) # Reshape the data into the shape the network expects
train_images = train_images.astype('float32') / 255 # Normalize the data to [0, 1]
test_images = test_images.reshape((10000, 28*28))
test_images = test_images.astype('float32') / 255

# Categorically encode the labels
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# %%
# Build the network
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras import initializers
import numpy as np
import tensorflow as tf
import json

# Set random seed
np.random.seed(0)

def leaky_relu(z, alpha=0.01):
    return tf.maximum(alpha*z, z)

def sigmoid(z):
    return 1 / (1 + tf.exp(-z))

def classifier_model(layer_size, num_layers, activation, input_dim, dropout_rate=0.2):
    network = models.Sequential()

    # Add first layer
    network.add(layers.Dense(layer_size, 
                             input_dim=input_dim, 
                             activation=activation, 
                             kernel_initializer=initializers.RandomNormal(stddev=0.01)))

    for _ in range(num_layers):
        network.add(layers.Dense(layer_size, 
                                 activation=activation, 
                                 kernel_initializer=initializers.RandomNormal(stddev=0.01)))
        network.add(BatchNormalization())
        network.add(layers.Dropout(dropout_rate))
    network.add(layers.Dense(10, activation='softmax')) # Add the output layer

    # Compile the network
    network.compile(optimizer='rmsprop',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
    
    return network

# %% [markdown]
# Create the networks

# %%
model_1 = classifier_model(
    layer_size=256,
    num_layers=50, 
    input_dim=28*28,
    activation=leaky_relu,
    )

model_2 = classifier_model(
    layer_size=256,
    num_layers=20, 
    input_dim=28*28,
    activation=leaky_relu,
    )

model_3 = classifier_model(
    layer_size=256,
    num_layers=10, 
    input_dim=28*28,
    activation=sigmoid,
    )

model_4 = classifier_model(
    layer_size=256,
    num_layers=5, 
    input_dim=28*28,
    activation=sigmoid,
    )

models = [model_1, model_2, model_3, model_4]

# %% [markdown]
# Train (or load) the models

# %%
histories = []

for model in models:
    with tf.device('/gpu:4'):
        history = model.fit(train_images, 
                            train_labels, 
                            epochs=20, 
                            batch_size=128, 
                            validation_data=(test_images, test_labels),
        )
    histories.append(history.history)

# Save the histories
with open('MNIST/histories.json', 'w') as f:
    json.dump(histories, f)

# %%
import matplotlib.pyplot as plt

for i, history in enumerate(histories):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history['loss'], label='Training Loss')
    ax1.plot(history['val_loss'], linestyle='--', label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'Model {i+1} Loss')
    ax1.legend()

    ax2.plot(history['accuracy'], label='Training Accuracy')
    ax2.plot(history['val_accuracy'], linestyle='--', label='Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title(f'Model {i+1} Accuracy')
    ax2.legend()

    plt.savefig(f'MNIST/model_{i+1}.png')

    plt.show()



import tensorflow as tf
from tensorflow.keras import layers, models

'''
The purpose of this file is to train a CNN on the MNIST dataset for predicting
handwritten digits.
'''

# Load MNIST dataset
(training_images, training_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

#Normalize data from [0, 255[] to [0, 1] - speeds up training
training_images = training_images / 255.0
test_images = test_images / 255.0

#Conv2D expects a third channel for greyscale
training_images = training_images[..., tf.newaxis]
test_images = test_images[..., tf.newaxis]

#Create a sequential model (stacked layers)
model = models.Sequential([
    #32 filters, 3x3 kernels, ReLU activation (rectified linear unit)
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    #Shrinks feature maps by half
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(training_images, training_labels, epochs=5, validation_data=(test_images, test_labels))

model.save('mnist_model.keras')

#99% accuracy
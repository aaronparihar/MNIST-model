import tensorflow as tf
from tensorflow.keras import layers, models

'''
The purpose of this file is to train a CNN on the MNIST dataset for predicting
handwritten digits.
'''

# Load MNIST dataset
(training_images, training_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# Normalize data from [0, 255[] to [0, 1] - speeds up training
training_images = training_images / 255.0
test_images = test_images / 255.0

# Conv2D expects a third channel for greyscale
training_images = training_images[..., tf.newaxis]
test_images = test_images[..., tf.newaxis]

# Create a sequential model (stacked layers)
model = models.Sequential([
    # 32 filters, 3x3 kernels, ReLU activation (rectified linear unit)
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    # Down-sampling feature maps to condense information
    layers.MaxPooling2D((2, 2)),
    # Perform convolution on the pooled feature map to learn shapes, edges, etc.
    layers.Conv2D(64, (3, 3), activation='relu'),
    # Ensure model focuses on most important features by down-sampling
    layers.MaxPooling2D((2, 2)),
    # Reshape feature maps into a vector
    layers.Flatten(),
    # Create fully connected layers with ReLU (function that allows learning complex patterns)
    layers.Dense(64, activation='relu'),
    # Softmax takes previous layer, maps probabilities to 10 classes (0-9)
    layers.Dense(10, activation='softmax')
])

# ADAptive Movement estimation - update weights 
model.compile(optimizer='adam',
              # Loss function, this one is best for integer labels
              loss='sparse_categorical_crossentropy',
              # Want to track accuracy of model
              metrics=['accuracy'])

# Train model over 5 epochs (5 passes of training dataset)
model.fit(training_images, training_labels, epochs=5, validation_data=(test_images, test_labels))

# Save to a file
model.save('mnist_model.keras')

#Outputs 99% accuracy on test data
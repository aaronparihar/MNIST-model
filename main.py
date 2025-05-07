import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist

'''
The purpose of this file is to provide an interface for users to interact
with the model. Users may run this, select a digit to test, and visually
confirm the prediction of the model.
'''

# Load MNIST dataset
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()


# Normalize data from [0, 255] to [0, 1] - matches format model was trained on
training_images = training_images / 255.0
test_images = test_images / 255.0

# Add a channel for grayscale (Conv2D expects 3D input)
training_images = training_images[..., tf.newaxis]
test_images = test_images[..., tf.newaxis]

# Load the trained model
model = tf.keras.models.load_model('mnist_model.keras')

# Get user input for the digit they want to test
try:
    digit = int(input("Enter the digit you want to test (0-9): "))
except:
    print("Invalid input: please enter an integer")
    exit()

# Check if input is valid
if digit < 0 or digit > 9:
    print("Invalid input: please enter a digit between 0 and 9.")
    exit()

# Find all indices where the label matches the input digit
indices = np.where(training_labels == digit)[0]

# Randomly select an index from those where the label is the chosen digit
random_index = np.random.choice(indices)

# Get the corresponding image and label
image = training_images[random_index]
label = training_labels[random_index]

# Prepare the image for prediction
image_for_prediction = np.expand_dims(image, axis=0)  # Add batch dimension

# Predict the digit using the trained model
prediction = model.predict(image_for_prediction)
predicted_digit = np.argmax(prediction)
confidence = np.max(prediction)

# Display the output
plt.imshow(image, cmap='gray')
plt.title(f"Digit: {label}\nNormalized image:")
plt.xlabel(f"Predicted: {predicted_digit} (Confidence: {confidence * 100:.2f}%)")
plt.show()
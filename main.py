import tensorflow as tf
import numpy as np
import cv2

# Load the pre-trained model
model = tf.keras.models.load_model('model.h5')

# Define a list of class labels
class_labels = ['cat', 'dog']

catIndex = 0
dogIndex = 1

# Prompt the user to enter the image path
image_path = input("Enter the path of the image file: ")

# Load the image
image = cv2.imread(image_path)

# resize the image
image = cv2.resize(image, (224, 224))

# convert the image to float
image = image.astype('float32') / 255.0

#  process the image
image = np.expand_dims(image, axis=0)

# Use the model to make predictions
predictions = model.predict(image) 

# [[catProbability, dogProbability]]

catScore = predictions[0][catIndex]
dogScore = predictions[0][dogIndex]

print("cat score: ", catScore * 100)
print("dog score: ", dogScore * 100)
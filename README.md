## IEEE-Compition

## Egyptian Landmarks Recognition Model

## 1. Project Overview

This project focuses on using a pre-trained ResNet50 model to classify Egyptian landmarks from user-uploaded images. The model is designed to accurately identify landmarks and provide relevant information about each.

## 2. Model Overview
The model used is ResNet50 with custom modifications to the top layers. The model takes images as input and outputs the predicted landmark category.

ResNet50: A deep convolutional neural network used for image classification.

Dense Layers: Added to perform final classification of the landmarks.

Softmax: Used as the activation function in the output layer to predict the probabilities of different landmark categories.

## 3. Model Training
# Data
The dataset consists of images of Egyptian landmarks, split into training and test sets.

# Preprocessing
All images were resized to 456x456 pixels. Data was normalized to have values between 0 and 1.

# Model
ResNet50 was used with its top classification layers removed, and custom Dense layers were added for final classification.

## 4. Architecture
ResNet50: The model consists of 50 layers, including convolutional layers and max-pooling layers.

Additional layers:

Flatten Layer: Converts the output of the convolutional layers to a 1D vector.

Dense Layer: A fully connected layer with 256 units and ReLU activation.

Output Layer: A fully connected layer with units corresponding to the number of landmark categories, with Softmax activation.

## 5. Performance
Accuracy: The model achieved 97.92% accuracy on the test set.

Other Metrics: Performance was evaluated using Precision, Recall, and F1-score to provide a comprehensive evaluation of the model.

## 6. How to Use
To upload an image and classify it using the model:

from tensorflow.keras.preprocessing import image import numpy as np

Load and preprocess image
img = image.load_img('path_to_image', target_size=(456, 456)) img_array = image.img_to_array(img) img_array = img_array / 255.0 img_array = np.expand_dims(img_array, axis=0)

Predict the landmark
prediction = model.predict(img_array) predicted_class = np.argmax(prediction)

After uploading the image, the model will return the predicted landmark along with relevant information.

## 7. Challenges and Future Improvements
# Challenges

Collecting sufficient data for all landmarks was one of the primary challenges.

Images with unclear or unusual angles of landmarks were difficult for the model to classify accurately.

# Future Improvements
Increase the dataset size by adding more images, especially for landmarks with less representation.

Use Data Augmentation techniques to increase data variety and improve model robustness.

## 8. Conclusion
The model enhances the user experience by enabling easy and effective exploration of Egyptian landmarks. With additional improvements in the future, the application will become even more accurate and interactive.

# Convolutional Deep Neural Network for Digit Classification

## AIM

To Develop a convolutional deep neural network for digit classification and to verify the response for scanned handwritten images.

## Problem Statement and Dataset
Digit classification and to verify the response for scanned handwritten images.

The MNIST dataset is a collection of handwritten digits. The task is to classify a given image of a handwritten digit into one of 10 classes representing integer values from 0 to 9, inclusively. The dataset has a collection of 60,000 handwrittend digits of size 28 X 28. Here we build a convolutional neural network model that is able to classify to it's appropriate numerical value.

## Neural Network Model

![387453381-86314a03-26c6-41b8-b442-9e68ad2600f4](https://github.com/user-attachments/assets/91fd6f7b-3dbc-448f-8159-c262243398b0)

## DESIGN STEPS

## STEP 1:
Import tensorflow and preprocessing libraries.

## STEP 2:
Download and load the dataset

## STEP 3:
Scale the dataset between it's min and max values

## STEP 4:
Using one hot encode, encode the categorical values

## STEP 5:
Split the data into train and test

## STEP 6:
Build the convolutional neural network model

## STEP 7:
Train the model with the training data

## STEP 8:
Plot the performance plot

## STEP 9:
Evaluate the model with the testing data

## STEP 10:
Fit the model and predict the single input


## PROGRAM

### Name:BALAJI J
### Register Number:212221243001


```
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import utils
import pandas as pd
from sklearn.metrics import classification_report,confusion_matrix
from tensorflow.keras.preprocessing import image

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train.shape

X_test.shape

single_image= X_train[0]

single_image.shape

plt.imshow(single_image,cmap='gray')

y_train.shape

X_train.min()

X_train.max()

X_train_scaled = X_train/255.0
X_test_scaled = X_test/255.0

X_train_scaled.min()

X_train_scaled.max()

y_train[0]

y_train_onehot = utils.to_categorical(y_train,10)
y_test_onehot = utils.to_categorical(y_test,10)

type(y_train_onehot)

y_train_onehot.shape

single_image = X_train[700]
plt.imshow(single_image,cmap='gray')
print("YUVASAKTHI N.C \n212222240120")

y_train_onehot[500]

X_train_scaled = X_train_scaled.reshape(-1,28,28,1)
X_test_scaled = X_test_scaled.reshape(-1,28,28,1)

model = keras.Sequential()
model.add(layers.Input(shape=(28,28,1)))
model.add(layers.Conv2D(filters=16,kernel_size=(3,3),activation='relu'))
model.add(layers.Conv2D(filters=32,kernel_size=(3,3),activation='relu'))
model.add(layers.MaxPool2D(pool_size=(2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(units=32,activation='relu'))
model.add(layers.Dense(units=10,activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy', # Loss function for multi-class classification
              optimizer='adam', # A popular optimization algorithm
              metrics=['accuracy']) # Metric to evaluate model performance

model.fit(X_train_scaled ,y_train_onehot, epochs=5,
          batch_size=64,
          validation_data=(X_test_scaled,y_test_onehot))

metrics = pd.DataFrame(model.history.history)

metrics.head()

metrics[['accuracy','val_accuracy']].plot()
print("\nBALAJI \n212221243001")

metrics[['loss','val_loss']].plot()
print("\nBALAJI \n212221243001")

x_test_predictions = np.argmax(model.predict(X_test_scaled), axis=1)

print(confusion_matrix(y_test,x_test_predictions))
print("\nYUVASAKTHI N.C \n212222240120")

print(classification_report(y_test,x_test_predictions))
print("\nBALAJI \n212221243001")

img = image.load_img('imagefour.png')

type(img)

img = image.load_img('imagefour.png')
img_tensor = tf.convert_to_tensor(np.asarray(img))
img_28 = tf.image.resize(img_tensor,(28,28))
img_28_gray = tf.image.rgb_to_grayscale(img_28)
img_28_gray_scaled = img_28_gray.numpy()/255.0

x_single_prediction = np.argmax(
    model.predict(img_28_gray_scaled.reshape(1,28,28,1)),
     axis=1)

print(x_single_prediction)

plt.imshow(img_28_gray_scaled.reshape(28,28),cmap='gray')
print("YUVASAKTHI N.C \n212222240120")

img_28_gray_inverted = 255.0-img_28_gray
img_28_gray_inverted_scaled = img_28_gray_inverted.numpy()/255.0

x_single_prediction = np.argmax(
    model.predict(img_28_gray_inverted_scaled.reshape(1,28,28,1)),
     axis=1)

print(x_single_prediction)
```

## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot

![367154627-e9779c84-0a95-4c4c-902c-6a9b29068528](https://github.com/user-attachments/assets/22bb3083-26e2-4e3b-b66e-2d65caa3755d)

![367154892-0a17caad-d1a0-4e9a-ae8d-1461acc6f81e](https://github.com/user-attachments/assets/d82ab7a5-091d-4f1e-a62c-57c226a641cc)


### Classification Report

![367154767-5f3b96f3-2f5b-4934-81ff-038df5b3e820](https://github.com/user-attachments/assets/5ed8aa89-c959-4292-b3c2-6b56961aead9)

### Confusion Matrix

![367154786-7dd49ff5-85c9-41b7-800a-919f4df64315](https://github.com/user-attachments/assets/08934f69-d27e-4eec-a72e-8556fc51e25b)


### New Sample Data Prediction

![367154936-1f8d2594-3c2f-4a44-97dc-1e53c4c354ee](https://github.com/user-attachments/assets/9f5f4663-312a-42fc-bf82-2b5145d95408)
![D45](https://github.com/user-attachments/assets/63a7a47d-2208-4313-99d1-81844f9fa56b)


## RESULT

A convolutional deep neural network for digit classification and to verify the response for scanned handwritten images is developed successfully.


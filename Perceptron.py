import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout

# Define paths to the train, validation, and test data folders

test_data_dir = os.path.join("..", "DRDC_data", "preprocessed", "test_images")
train_data_dir = os.path.join("..", "DRDC_data", "preprocessed", "train_images")
val_data_dir = os.path.join("..", "DRDC_data", "preprocessed", "val_images")


#train_data_dir = 'C:/Users/aditi/Downloads/Project/DRDC_data/preprocessed/train_images'
#validation_data_dir = 'C:/Users/aditi/Downloads/Project/DRDC_data/preprocessed/val_images'
#test_data_dir = 'C:/Users/aditi/Downloads/Project/DRDC_data/preprocessed/test_images'


# Load labels from the Excel sheets as described in previous responses.
train_labels = pd.read_csv('train_1.csv')
validation_labels = pd.read_csv('valid.csv')

# Define a function to preprocess the labels
def preprocess_labels(labels):
    labels['Stage'] = labels['Stage'].apply(lambda x: 1 if x > 0 else 0)
    return labels

# Preprocess train and validation labels
train_labels = pd.read_csv('train_1.csv')
validation_labels = pd.read_csv('valid.csv')
train_labels = preprocess_labels(train_labels)
validation_labels = preprocess_labels(validation_labels)

# Create data generators for training, validation, and test data
train_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

batch_size = 32

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='binary')

validation_generator = validation_datagen.flow_from_directory(
    val_data_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='binary')

# Neural network model with one hidden layer
model = Sequential()
model.add(Flatten(input_shape=(224, 224, 3)))

# Hidden layer
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

# Output layer
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Training the model
epochs = 10
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=len(validation_generator))

# Model Evaluation
validation_loss, validation_accuracy = model.evaluate(validation_generator)
print(f'Validation Accuracy (With Hidden Layer): {validation_accuracy * 100:.2f}%')

# Data Generator
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='binary')  

# Accuracy for testing images
test_loss, test_accuracy = model.evaluate(test_generator)

print(f'Test Accuracy: {test_accuracy * 100:.2f}%')

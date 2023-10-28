import os
current_dir = os.getcwd()
print("Current Working Directory:", current_dir)
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.callbacks import Callback

# Define paths to the train, validation, and test data folders

preprocess_dir = os.path.join("DRDC_data", "preprocessed")

test_data_dir = os.path.join(preprocess_dir, "test")
train_data_dir = os.path.join(preprocess_dir, "train")
val_data_dir = os.path.join(preprocess_dir, "validation")


#train_data_dir = 'C:/Users/aditi/Downloads/Project/DRDC_data/preprocessed/train_images'
#validation_data_dir = 'C:/Users/aditi/Downloads/Project/DRDC_data/preprocessed/val_images'
#test_data_dir = 'C:/Users/aditi/Downloads/Project/DRDC_data/preprocessed/test_images'


# Load labels from the Excel sheets as described in previous responses.
train_labels = pd.read_csv(os.path.join(preprocess_dir,'train_1.csv'))
validation_labels = pd.read_csv(os.path.join(preprocess_dir,'valid.csv'))

# Define a function to preprocess the labels

def preprocess_labels(labels):
    labels['diagnosis'] = labels['diagnosis'].apply(lambda x: 1 if x > 0 else 0)
    return labels

# Preprocess train and validation labels
train_labels = preprocess_labels(train_labels)
validation_labels = preprocess_labels(validation_labels)



# Create data generators for training, validation, and test data
#train_datagen = ImageDataGenerator(rescale=1./255)

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

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


# Define a custom callback to stop training when accuracy exceeds 80%
class CustomEarlyStopping(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs['val_accuracy'] > 0.89:
            print("Validation accuracy exceeded 80%, stopping training.")
            self.model.stop_training = True

# Neural network model with one hidden layer
model = Sequential()
#model.add(Flatten(input_shape=(224, 224, 3)))

# Convolutional layers
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(BatchNormalization())

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(BatchNormalization())

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(BatchNormalization())

# Flatten the output for fully connected layers
model.add(Flatten())

# Fully connected layers
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))  # Add dropout to reduce overfitting
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))


# Hidden layer
# model.add(Dense(128, activation='relu'))
# model.add(Dense(64, activation='relu'))  # Add another hidden layer with ReLU activation
# model.add(Dense(32, activation='relu'))  # Add another hidden layer with ReLU activation
# model.add(Dense(64, activation='sigmoid'))
# model.add(Dropout(0.5))


# Output layer
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Training the model with the custom early stopping callback
epochs = 20  # Set a large number of epochs
custom_early_stopping = CustomEarlyStopping()
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=len(validation_generator),
    callbacks=[custom_early_stopping])

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

import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot as plt
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Conv2D
import os
from tensorflow.python.framework import graph_io
import json

image_size = 128 # All images will be resized to 224x224
batch_size = 12
train_dir = r'E:\Estek-AIProject\INTEL-SOOP\RaspberryPiFolder\Images\Training\combinedForTrainingColour'
validation_dir = r'E:\Estek-AIProject\INTEL-SOOP\RaspberryPiFolder\Images\Training\ValidationSet'
# Rescale all images by 1./255 and apply image augmentation
train_datagen = keras.preprocessing.image.ImageDataGenerator(
                rescale=1./255)

validation_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

# Flow training images in batches of 20 using train_datagen generator
train_generator = train_datagen.flow_from_directory(
                train_dir,  # Source directory for the training images
                target_size=(image_size, image_size),
                batch_size=batch_size,
                # Since we use binary_crossentropy loss, we need binary labels
                class_mode='binary')

print(train_generator.class_indices)
print(train_generator.classes)
labels = train_generator.class_indices
labels = dict((v,k) for k,v in labels.items())
jason = json.dumps(labels)
f = open('labels.json', 'w')
f.write(jason)
f.close()

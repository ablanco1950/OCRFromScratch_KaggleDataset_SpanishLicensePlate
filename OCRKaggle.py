# -*- coding: utf-8 -*-
"""
Created on Fri Dec 23 12:18:44 2022

@author: https://www.kaggle.com/code/preatcher/ocr-training
"""

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

#Changing the directory to data2 folder

import os
#os.chdir("../input/standard-ocr-dataset/data")
os.chdir("C:/archive/data")

#importing Necessary Modules

import tensorflow as tf
import datetime
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import regularizers
from tensorflow.keras import initializers
import matplotlib.pyplot as plt
import cv2

# Creating a Function to plot images
### defining some function to make our work easier
import matplotlib.pyplot as plt

# This function will plot images in the form of a grid with 1 row and 5 columns where images are placed in each column.
def plot_images(images_arr, imageWidth, imageHeight):
    fig, axes = plt.subplots(1, 5, figsize=(20, 20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img.reshape(imageWidth, imageHeight), cmap="gray")
        ax.axis("off")
    plt.tight_layout()
    plt.show()
    
# Some constants to be defined

batch_size = 32
epochs = 50
IMG_HEIGHT = 28
IMG_WIDTH = 28

# Image Data Generator
# We defined two data generators, one that augments the data
# to make our training more general and one that just scales
# and centers the data.

def preprocessing_fun(img):
#     print(img.shape)
#     print(img)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY, CV_8UC1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img,(3,3),0)
    img = img.reshape((28,28,1))
    thresh = cv2.adaptiveThreshold(img, 255, 1, 1, 11, 2)
    print(thresh.shape)
    

#     img = img.reshape((28,28,1))

#     img = np.where(img>140,1,0)
#     img  = img/255
#     return img

augmented_image_gen = ImageDataGenerator(
        rescale = 1/255.0,
    rotation_range=2,
    width_shift_range=.1,
    height_shift_range=.1,
    zoom_range=0.1,
    shear_range=2,
    brightness_range=[0.9, 1.1],
    validation_split=0.2,
   
   )

normal_image_gen = ImageDataGenerator(
    rescale = 1/255.0,
    validation_split=0.2,
  
   )

#Using Data Generator generate batches to train our model

train_data_gen = augmented_image_gen.flow_from_directory(batch_size=batch_size,
                                                     directory="./training_data",
                                                     color_mode="grayscale",
                                                     shuffle=True,
                                                     target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                     class_mode="categorical",
                                                     seed=65657867,
                                                     subset='training')
val_data_gen = normal_image_gen.flow_from_directory(batch_size=batch_size,
                                                     directory="./testing_data",
                                                     color_mode="grayscale",
                                                     shuffle=True,
                                                     target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                     class_mode="categorical",
                                                     seed=65657867,
                                                     subset='validation')
# samples

sample_training_images, _ = next(train_data_gen)
plot_images(sample_training_images[:7], IMG_WIDTH, IMG_HEIGHT)

# Defining our Sequential model

# model = Sequential([
#     Conv2D(16, 3, 
#            padding='same',
#            activation='relu',
#            kernel_regularizer=regularizers.l2(0.0001),
#            input_shape=(IMG_HEIGHT, IMG_WIDTH , 1)),
#     MaxPooling2D(),
#     Dropout(0.2),
#     Flatten(),
#     Dense(
#         50,
#         activation='relu',
#         kernel_regularizer=regularizers.l2(0.0001)
#     ),
#     Dropout(0.2),
#     Dense(36, activation='softmax')
# ])

from tensorflow.keras.optimizers import SGD
# define cnn model
def define_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    #model.add(Dense(1000, activation='relu', kernel_initializer='he_uniform'))
    # https://www.geeksforgeeks.org/weight-initialization-techniques-for-deep-neural-networks/
   
    #initializer = tf.keras.initializers.RandomUniform(minval=0.,maxval=1.)
    #model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
    initializer = tf.keras.initializers.Zeros()
    model.add(Dense(100, activation='relu'))
    model.add(Dense(36, activation='softmax', kernel_initializer=initializer))
    return model
#     # compile model
# 	opt = SGD(lr=0.01, momentum=0.9)
# 	model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
# 	return model
model = define_model()

import tensorflow
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint, ReduceLROnPlateau

#Prepare call backs

EarlyStop_callback = tensorflow.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
checkpoint = ModelCheckpoint('/kaggle/working/checkpoint',
                             monitor = 'val_loss',mode = 'min',save_best_only= True)
lr = ReduceLROnPlateau(monitor = 'val_loss',factor = 0.5,patience = 3,min_lr = 0.00001)
my_callback=[EarlyStop_callback,checkpoint]

# Actual training of our model
 
model.compile(optimizer=SGD(lr=0.01, momentum=0.9),
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=['categorical_accuracy'])

history = model.fit_generator(
    train_data_gen,
    steps_per_epoch=train_data_gen.samples // batch_size,
    epochs=42,
    validation_data=val_data_gen,
    validation_steps=val_data_gen.samples // batch_size,
    #callbacks = my_callback
    )

model.save("ModelOCRKaggle42Epoch.h5")
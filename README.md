OCR from scratch using Kaggle dataset dwonloaded from https://www.kaggle.com/code/preatcher/ocr-training  applied to the case of Spanish car license plates or any other with format NNNNAAA. The hit rate is lower than that achieved by pytesseract: in a test with 21 images, 15 hits are reached while with pytesseract the hits are 17 https://github.com/ablanco1950/LicensePlate_Labeled_MaxFilters.

Requirements:

have the packages installed that allow:

import numpy

import tensorflow

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Conv2D

from tensorflow.keras.layers import MaxPooling2D

from tensorflow.keras.layers import Flatten

from tensorflow.keras.layers import Dense, Dropout

import cv2

Download from https://www.kaggle.com/datasets/preatcher/standard-ocr-dataset the archive.zip file, unzip it.

In the download directory you should find the downloaded test6Training.zip (roboflow.com) and must unzip folder: test6Training with all its subfolders, containing the images for the test and its labels. This directory must be in the same directory where is the program GetNumberSpanishLicensePlate_OCRKaggle_labels_MaxFilters.py ( unziping may create two directories with name test6Training and the images may not be founded when executing it, it would be necessary copy of inner directory test6Training in the same directory where is  the mentioned  program GetNumberSpanishLicensePlate_OCRKaggle_labels_MaxFilters.py)

Operative:

Execute the program:

OCRKaggle.py

that creates the model ModelOCRKaggle25Epoch.h5 

It requires that the file directory, with the kaggle characters used to train the model be in C:, although its location can be changed by altering line 30
 of OCRKaggle.py.

this model  is created in the archive/data directory and must be passed to the program's execution directory, the directory where is GetNumberSpanishLicensePlate_OCRKaggle_labels_MaxFilters.py

Execute:

GetNumberSpanishLicensePlate_OCRKaggle_labels_MaxFilters.py

That uses the model ModelOCRKaggle25Epoch.h5  create in the step before

Each car license plate appears on the screen with the text that could have been recognized from the image and the final result assigning the car license plate that has been recognized the most times.

As output, the LicenseResults.txt file is also obtained with the relation between true license plate and predicted license plate.

Observations:


The OCRKaggle.py program, which is a copy of the one found at https://www.kaggle.com/code/preatcher/ocr-training, 

with the following changes:

    The number of filters is reduced to 8 from 32
    
    the kernel is increased to (5,5) from (3,3)
    
    the Dense with activation relu is increased to 250 from 100
    
    the kernel_initializer of last dense with activation softmax is set to
    initializer instead of he_uniform

    To avoid that the values obtained in the CNN model vary from one execution to another of OCRKaggle.py, 
    the weights have to be initialized to a fixed value by the added instruction
    (https://stackoverflow.com/questions/46407457/error-in-creating-custom-initializer-using-get-variable-with-keras)
    initializer =tf.keras.initializers.glorot_normal()

References:

https://www.kaggle.com/code/preatcher/ocr-training

https://www.kaggle.com/datasets/preatcher/standard-ocr-dataset

https://www.roboflow.com

https://www.geeksforgeeks.org/weight-initialization-techniques-for-deep-neural-networks/
   
https://github.com/ablanco1950/LicensePlate_Labeled_MaxFilters

https://github.com/ablanco1950/OCRFromScratch_Chars74K_SpanishLicensePlate

https://stackoverflow.com/questions/46407457/error-in-creating-custom-initializer-using-get-variable-with-keras

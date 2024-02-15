
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.utils import plot_model
from keras.layers import Conv2D, MaxPool2D,Dropout, Dense, Input, concatenate,GlobalAveragePooling2D, AveragePooling2D,Flatten

class VGG16Sequential:
    def __init__(self, classes,input_shape=(224, 224, 3)):
        self.input_shape=input_shape
        self.model=self.build_model()
        self.classes=classes
    def build_model(self):
        model = Sequential()

        model.add(Conv2D(64, (3, 3), padding='SAME', activation='relu', input_shape=self.input_shape))
        model.add(Conv2D(64, (3, 3), padding='SAME', activation='relu'))
        model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Conv2D(128, (3, 3), padding='SAME', activation='relu'))
        model.add(Conv2D(128, (3, 3), padding='SAME', activation='relu'))
        model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(Conv2D(256, (3, 3), padding='SAME', activation='relu'))
        model.add(Conv2D(256, (3, 3), padding='SAME', activation='relu'))
        model.add(Conv2D(256, (3, 3), padding='SAME', activation='relu'))
        model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(Conv2D(512, (3, 3), padding='SAME', activation='relu'))
        model.add(Conv2D(512, (3, 3), padding='SAME', activation='relu'))
        model.add(Conv2D(512, (3, 3), padding='SAME', activation='relu'))
        model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(Conv2D(512, (3, 3), padding='SAME', activation='relu'))
        model.add(Conv2D(512, (3, 3), padding='SAME', activation='relu'))
        model.add(Conv2D(512, (3, 3), padding='SAME', activation='relu'))
        model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(Flatten())

        model.add(Dense(256, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(10, activation='softmax'))

        return model
    
    
def VGG16(classes=10):
    vgg16_sequential = VGG16Sequential(classes)
    model = vgg16_sequential.model
    return model
model=VGG16(classes=10)
 


import tensorflow as tf
from copy import deepcopy
import numpy as np

class DataSet(object):
    def __init__(self, load_fun, target, classes, augmentation=True):

        
        (self.x_train, self.y_train), (self.x_test, self.y_test) = load_fun
        self.x_train = self.x_train.astype("float32") / 255.
        self.x_test = self.x_test.astype("float32") / 255.
        print(self.x_train[0].shape)
        print(type(self.x_train[0]))
        # self.x_train, self.y_train = self.x_train[:10], self.y_train[:10]
        # self.x_test, self.y_test = self.x_test[:10], self.y_test[:10]
        print("reshaping of training images started")
        self.x_train= self.reshape_images(self.x_train)
        print("reshaping of testing images started")
        self.x_test= self.reshape_images(self.x_test)
        print("reshaping ended")
        print(self.x_train[0].shape)
        print(type(self.x_train[0]))


        self.train_samples = self.x_train.shape[0]
        self.target = target
        self.classes = classes
        self.augmentation = augmentation
        print("making of deepcopy")
        self.x_train_poison, self.y_train_poison, self.x_test_poison, self.y_test_poison, =\
            deepcopy(self.x_train), deepcopy(self.y_train), deepcopy(self.x_test), deepcopy(self.y_test)
        print("deep copy making done")
        

        #generators are returned here
        print("preprocess started")
        self.image_gen = self.preprocess()
        print("preprocess_poison started")
        self.image_gen_poison = self.preprocess_poison()
        print("preprocess_ended")

        #initialization k baad we have :
        #x_train, y_train, x_test, y_test, x_train_poison, y_train_poison, x_test_poison, y_test_poison 
        #=> poison waale saare y ki value = target k equal h

        #ye toh initialization ka hua : but jab ds_data call hoga toh:
        #ds_train, ds_train_backdoor, ds_test, ds_test_backdoor, _ = dataset.ds_data(batch_size, backdoor=True)
        #=> 
    def reshape_images(self, images):
        resized_images = tf.image.resize(images, size=(224, 224), method=tf.image.ResizeMethod.BILINEAR)
        resized_images_np = np.array(resized_images)
        return resized_images_np


    def preprocess(self):
        if self.augmentation:
            image_gen_train = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                horizontal_flip=True,
            )
        else:
            image_gen_train = tf.keras.preprocessing.image.ImageDataGenerator()


        image_gen_train.fit(self.x_train)
        
        return image_gen_train

    def preprocess_poison(self):
        image_size = self.x_train_poison.shape[1]
        pattern_a = int(image_size * 0.75)
        pattern_b = int(image_size * 0.9375)

        for i in range(len(self.x_train_poison)):
            self.x_train_poison[i, pattern_a:pattern_b, pattern_a:pattern_b] = 1
            # print("AKANKSHA")
            self.y_train_poison[i] = self.target
        for i in range(len(self.x_test_poison)):
            self.x_test_poison[i, pattern_a:pattern_b, pattern_a:pattern_b] = 1
            self.y_test_poison[i] = self.target

        if self.augmentation:
            image_gen_poison = tf.keras.preprocessing.image.ImageDataGenerator()
        else:
            image_gen_poison = tf.keras.preprocessing.image.ImageDataGenerator()
        
        image_gen_poison.fit(self.x_train_poison)
        return image_gen_poison

    def ds_data(self, batch_size):
        ds_train = self.image_gen.flow(
            self.x_train, self.y_train, batch_size=batch_size
        )

        ds_train_backdoor = self.image_gen_poison.flow(
                self.x_train_poison, self.y_train, batch_size=batch_size
            )
        ds_train_backdoor_y_poison = self.image_gen_poison.flow(
                self.x_train_poison, self.y_train_poison, batch_size=batch_size
            )
        

        ds_test = tf.data.Dataset.from_tensor_slices((self.x_test, self.y_test)) \
            .batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

        ds_test_backdoor = tf.data.Dataset.from_tensor_slices((self.x_test_poison, self.y_test)) \
            .batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
        ds_test_backdoor_y_poison = tf.data.Dataset.from_tensor_slices((self.x_test_poison, self.y_test_poison)) \
            .batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

    

        return ds_train, ds_train_backdoor, ds_train_backdoor_y_poison, ds_test, ds_test_backdoor, ds_test_backdoor_y_poison


def Cifar10(target=0):
    cifar10 = DataSet(load_fun=tf.keras.datasets.cifar10.load_data(), target=target, classes=10)
    return cifar10

# -*- coding: utf-8 -*-
"""
Created on Mon May 11 08:37:58 2020

@author: krish
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import kerastuner as kt
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers
from kerastuner.tuners import RandomSearch
from kerastuner.engine.hyperparameters import HyperParameters

fashion_mnist = keras.datasets.fashion_mnist

(x, y), (val_x, val_y) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

 # reshape dataset to have a single channel
z = x.reshape((x.shape[0], 28, 28, 1))
val_x = val_x.reshape((val_x.shape[0], 28, 28, 1))
# one hot encode target values
y = to_categorical(y).astype('int')
val_y = to_categorical(val_y).astype('int')
   

# Preprocess the data (these are Numpy arrays)
x = x.reshape(x.shape).astype('float32') / 255
val_x = val_x.reshape(val_x.shape).astype('float32') / 255





print(x.shape)
print(val_x.shape)

print(y)


plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x[i], cmap=plt.cm.binary)
    plt.xlabel(y[i])
plt.show()


image_rows = 28
image_cols = 28
batch_size = 512
image_shape = (image_rows,image_cols,1) 


x = x.reshape(x.shape[0], *image_shape)
val_x = val_x.reshape(val_x.shape[0], *image_shape)



model = tf.keras.Sequential()
#Adding the first input layer 

model.add(tf.keras.layers.Conv2D(filters=64, 
                                 kernel_size=(3,3),
                                 padding='same',
                                 activation='relu',
                                 input_shape=image_shape))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.MaxPooling2D())
model.add(tf.keras.layers.Dropout(0.25))

#Adding the second input layer
model.add(tf.keras.layers.Conv2D(filters=64,
                                 kernel_size=(3, 3),
                                 padding='same',
                                 activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.MaxPooling2D())
model.add(tf.keras.layers.Dropout(0.25))

#Adding the third input layer
model.add(tf.keras.layers.Conv2D(filters=64,
                                 kernel_size=(3, 3),
                                 padding='same',
                                 activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.MaxPooling2D())
model.add(tf.keras.layers.Dropout(0.25))


model.add(tf.keras.layers.Flatten())

#Adding the first Dense Layer
model.add(tf.keras.layers.Dense(500, activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dropout(0.25))

#Adding the second Dense Layer
model.add(tf.keras.layers.Dense(10, activation='softmax'))

model.compile(loss=tf.keras.losses.binary_crossentropy,
              optimizer=tf.keras.optimizers.Adam(lr=0.01),
              metrics=['accuracy'])

# Take a look at the model summary
model.summary()

   
#WE can use the model.compile() to configure the learning process bfore trainig model
''' model.compile(loss='binary_crossentropy', 
              optimizer=tf.keras.optimizers.Adam(lr=0.001),
            metrics=['accuracy']
             )
 ''' 
history = model.fit(x,y,
             batch_size=256, 
             epochs=10, 
             verbose=1,
             validation_data=(val_x,val_y)
             )

test_loss, test_accuracy = model.evaluate(val_x, val_y, verbose=2)

# Print test accuracy
print('\n', 'Test Loss:', test_loss)
print('\n', 'Test accuracy:', test_accuracy)


'''


tuner = RandomSearch(   
    model,
    objective= kt.Objective('val_accuracy', 'max'),
    max_trials=5,
    executions_per_trial=10,
    directory=tf.io.gfile.makedirs('myDir'),
    project_name='Fashion',
    overwrite=True)



tuner.search(x=x,
             y=y,
             epochs=3,
             validation_split=0.1)

tuner.results_summary()
'''

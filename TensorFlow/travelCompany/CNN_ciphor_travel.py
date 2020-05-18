# -*- coding: utf-8 -*-
"""
Created on Thu May 14 10:22:53 2020

@author: krish
"""

"""
DESCRIPTION

Companies in the travel and transportation industries are 
discovering many inventive uses for image classification. For instance, 
Singapore’s Changi Airport is equipping its new terminal with 
computer vision to leverage facial recognition during check-in, security, 
and departure processes. To achieve the same you are provided 
with a CIFAR-10 dataset. It consists of 60000, 32x32 color 
images in 10 classes.

Objective: Build a neural network-based classification model to 
recognize characters using the following metrics:
● ReLu as an activation function
● SGD as optimizer
● 2D convolutional and max pooling layer
"""

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import os
from matplotlib import pyplot
from numpy import mean
from numpy import std


batch_size = 32
num_classes = 10
epochs = 100
data_augmentation = True
num_predictions = 20
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keras_cifar10_trained_model.h5'

# load train and test dataset
def load_dataset():
    # load dataset
    (trainX, trainY), (testX, testY) = cifar10.load_data()
    print(trainX.shape)
    # reshape dataset to have a single channel
    trainX = trainX.reshape((trainX.shape[0], 32, 32, 3))
    testX = testX.reshape((testX.shape[0], 32, 32, 3))
    # one hot encode target values
    trainY = to_categorical(trainY)
    testY = to_categorical(testY)
   
    for i in range(9):
    # define subplot
        pyplot.subplot(350 + 1 + i)
        # plot raw pixel data
        pyplot.imshow(trainX[i])
    # show the figure
    pyplot.show()
    return trainX, trainY, testX, testY

# scale pixels
def prep_pixels(train, test):
    # convert from integers to floats
    train_norm = train.astype('float32')
    test_norm = test.astype('float32')
    # normalize to range 0-1
    train_norm = train_norm / 255.0
    test_norm = test_norm / 255.0
    # return normalized images
   
    return train_norm, test_norm

# define cnn model
def define_model(train_ix):
    model = Sequential()
    #Conv first pass
    model.add(Conv2D(32,
                     (3, 3),
                     padding='same',
                     activation='relu',
                     kernel_initializer='he_uniform',
                     input_shape = (32, 32, 3)))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
      #Conv third pass
    model.add(Conv2D(64,
                     (3, 3),
                     padding='same',
                     activation='relu',
                     kernel_initializer='he_uniform',
                     ))
 

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Flatten())
    
    model.add(Dense(512, activation='relu',
                    kernel_initializer='he_uniform'))
    model.add(Dropout(0.5))
    
    model.add(Dense(num_classes, activation='softmax'))
   
    # compile model
    opt = SGD(lr=0.01, momentum=0.9)
    
    model.compile(optimizer=opt,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    # printing the model summary
    print(model.summary())
    
    return model



# plot diagnostic learning curves
def summarize_diagnostics(histories):
    # plot loss
    pyplot.subplot(211)
    pyplot.title('Cross Entropy Loss')
    pyplot.plot(histories.history['loss'], color='blue', label='train')
    pyplot.plot(histories.history['val_loss'], 
              color='orange', label='test')
    # plot accuracy
    pyplot.subplot(212)
    pyplot.title('Classification Accuracy')
    pyplot.plot(histories.history['acc'], color='blue', label='train')
    pyplot.plot(histories.history['val_acc'], 
              color='orange', label='test')
    pyplot.show()


# summarize model performance
def summarize_performance(scores):
    # print summary
    print('Accuracy: mean=%.3f std=%.3f' 
       % ( mean(scores) *100, std(scores) *100 ) )
    # box and whisker plots of results
    pyplot.boxplot(scores)
    pyplot.show()
    

# run the test harness for evaluating a model
def run_test_harness():
    # load dataset
    trainX, trainY, testX, testY = load_dataset()
    
    # prepare pixel data
    trainX, testX = prep_pixels(trainX, testX)
    
    model = define_model(trainX)
    
    # fit model
    history = model.fit(trainX,trainY,epochs=2,batch_size=128,validation_data=(testX, testY),verbose=1)
  
    # evaluate model
    loss, acc = model.evaluate(testX, testY, verbose=1)

    print(acc)
  
    # learning curves
    summarize_diagnostics(history)
    
    # summarize estimated performance
    summarize_performance(acc)

# entry point, run the test harness
run_test_harness()
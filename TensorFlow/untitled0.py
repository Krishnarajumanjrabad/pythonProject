# -*- coding: utf-8 -*-
"""
Created on Tue May 12 10:43:48 2020

@author: krish
"""



import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

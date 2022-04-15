#!/usr/bin/env python
# coding: utf-8

# 1.Install  and import dependencies: OpenCV (for Andy); matplotlib for everyone

# In[1]:


import cv2 # with opencv we can access our webcam, which can be helpful for our application or data capturing.
import os
import random
import numpy as np
from matplotlib import pyplot as plt


# In[3]:


# Dependencies for model building- Tensorflow Functional API

from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten
import tensorflow as tf


# Create Folder Structure: Set up the paths and make the folders

# In[4]:


POS_PATH = os.path.join('data','positive')
NEG_PATH = os.path.join('data','negative')
ANC_PATH = os.path.join('data', 'anchor')


# In[11]:


os.makedirs(POS_PATH)
os.makedirs(NEG_PATH)
os.makedirs(ANC_PATH)


# 2.Collecting Data. I've already done that


# 3.Load and Preprocess Images

# Get Image Directories

# In[28]:


anchor = tf.data.Dataset.list_files(ANC_PATH+'\*.jpg').take(1500)
positive = tf.data.Dataset.list_files(POS_PATH+'\*.jpg').take(1500)
#negative = tf.data.Dataset.list_files(NEG_PATH+'\*.jpg').take(1500)


# In[29]:


dir_test = anchor.as_numpy_iterator()


# In[19]:


print(dir_test.next())


# Preprocessing- Scale and Resize

# In[30]:


def preprocess(file_path):
    #Read in image from file path
    byte_img = tf.io.read_file(file_path)
    #Load in the image 
    img = tf.io.decode_jpeg(byte_img)
    #Resizing the image to be 105x105x3
    img = tf.image.resize(img,(105,105))
    #Normalization or scale image between 0 and 1 
    img = img/255.0

    return img 


# In[31]:


img = preprocess('data\\anchor\\Amelie_Mauresmo_0016.jpg')


# In[32]:


plt.imshow(img)


# Create Labelled Dataset: Assign whether the both images are both faces or not

# In[25]:


positives = tf.data.Dataset.zip((anchor,positive,tf.data.Dataset.from_tensor_slices(tf.ones(len(anchor)))))
#negatives = tf.data.Dataset.zip((anchor,negative,tf.data.Dataset.from_tensor_slices(tf.ones(len(anchor)))))
#data = positives.concatenate(negatives)


# In[34]:


samples = positives.as_numpy_iterator()


# In[36]:


example = samples.next()
example


# Build Train and test sets

# In[39]:


def data_preparing(input_img, validation_img, label):
    return(preprocess(input_img),preprocess(validation_img),label)


# In[40]:


res = data_preparing(*example)
# The * is for the unpacking of the tupple


# In[42]:


plt.imshow(res[0])
# Plot of the anchor image 


# In[45]:


res[1]
# Real output: Array with numbers between 0 and 1 for each pixel.


# In[44]:


res[2]
# Since both are faces, I've assigned them as true or 1. Later, when the negative set is completed, we'll have another value 0 for false.


# In[46]:


# Build dataloader pipeline
positives = positives.map(data_preparing)
positives = positives.cache()


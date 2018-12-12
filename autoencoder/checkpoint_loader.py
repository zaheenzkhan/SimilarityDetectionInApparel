import os
import re
from glob import glob
import scipy.misc


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import random
import cv2
from PIL import Image
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.layers import fully_connected

num_inputs=258*258    #28x28 pixels
num_hid0=392*2
num_hid1=392
num_hid2=196
num_hid3=num_hid1
num_hid4=num_hid0
num_output=num_inputs
lr=0.001
actf=tf.nn.relu
# actf = tf.nn.sigmoid

num_epoch = 2
batch_size = 3
num_test_images = 10
image_shape=(1,258*258)
last=0

data_folder="./data"


X=tf.placeholder(tf.float32,shape=[None,num_inputs])
initializer=tf.variance_scaling_initializer()

with tf.name_scope('encode'):
  w0=tf.Variable(initializer([num_inputs,num_hid0]),dtype=tf.float32)
  w1=tf.Variable(initializer([num_hid0,num_hid1]),dtype=tf.float32)
  w2=tf.Variable(initializer([num_hid1,num_hid2]),dtype=tf.float32)
  b0=tf.Variable(tf.zeros(num_hid0))
  b1=tf.Variable(tf.zeros(num_hid1))
  b2=tf.Variable(tf.zeros(num_hid2))
  hid_layer0=actf(tf.matmul(X,w0)+b0)
  hid_layer1=actf(tf.matmul(hid_layer0,w1)+b1)
  hid_layer2=actf(tf.matmul(hid_layer1,w2)+b2)
with tf.name_scope('decode'):
  w3=tf.Variable(initializer([num_hid2,num_hid3]),dtype=tf.float32)
  w4=tf.Variable(initializer([num_hid3,num_hid4]),dtype=tf.float32)
  w5=tf.Variable(initializer([num_hid4,num_output]),dtype=tf.float32)
  b3=tf.Variable(tf.zeros(num_hid3))
  b4=tf.Variable(tf.zeros(num_hid4))
  b5=tf.Variable(tf.zeros(num_output))
  hid_layer3=actf(tf.matmul(hid_layer2,w3)+b3)
  hid_layer4=actf(tf.matmul(hid_layer3,w4)+b4)
  output_layer=actf(tf.matmul(hid_layer4,w5)+b5)


modelsaver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='encode'))

# init=tf.global_variables_initializer()


with tf.Session() as sess:
    modelsaver.restore(sess,"./model.ckpt")
    image = Image.open("./data/_123__the70sdiditbetter-t-shirt-kellygreen-258x258.jpg")
    image = image.convert("L")
    image = np.array(image, dtype=np.float32) / 255
    image = np.resize(image, image_shape)

    hidden = sess.run(hid_layer2,feed_dict={X: image})

print(len(image[0]))
print(len(hidden[0]))


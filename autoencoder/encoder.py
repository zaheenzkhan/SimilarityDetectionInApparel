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

num_epoch = 100
batch_size = 3
num_test_images = 10
image_shape=(258*258)
last=0

data_folder="./data"


def get_batches_fn(batch_size):
    """
    Create batches of training data
    :param batch_size: Batch Size
    :return: Batches of training data
    """
    # Grab image and label paths
    image_paths = glob(os.path.join(data_folder, '*.jpg'))

    # Shuffle training data
    if last==0:
        random.shuffle(image_paths)
    # Loop through batches and grab images, yielding each batch
    for batch_i in range(last, len(image_paths), batch_size):
        images = []
        for image_file in image_paths[batch_i:batch_i + batch_size]:
            # Re-size to image_shape

            image=Image.open(image_file)
            image = image.convert("L")
            image = np.array(image,dtype=np.float32)/255
            image = np.resize(image,image_shape)
            images.append(image)
        return images,images




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

loss=tf.reduce_mean(tf.square(output_layer-X))

optimizer=tf.train.AdamOptimizer(lr)
train=optimizer.minimize(loss)

modelsaver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='encode'))

init=tf.global_variables_initializer()



with tf.Session() as sess:
    sess.run(init)
    X_batch=[]
    results=[]
    for epoch in range(num_epoch):
        last=0
        num_batches = 66 // batch_size
        for iteration in range(num_batches):
            X_batch, y_batch = get_batches_fn(batch_size)
            sess.run(train, feed_dict={X: X_batch})
            last+=batch_size
        if epoch % 10 == 0:
            train_loss = loss.eval(feed_dict={X: X_batch})
            print("epoch {} loss {}".format(epoch, train_loss))
        # if epoch % 10 == 0:
        #     modelsaver.save(sess, './model.ckpt')
    results = output_layer.eval(feed_dict={X: X_batch})

    modelsaver.save(sess, './model.ckpt')

    npres = np.array(results[0], dtype=np.float64, copy=True)
    print(npres)
    img = np.array(np.resize(npres, (258, 258)),dtype=np.float32) * 255
    img = img.astype(np.uint8)
    print(img)
    im_pil = Image.fromarray(img)
    im_pil.save("img0.png", "PNG")


# # Comparing original images with reconstructions
#     f, a = plt.subplots(2, 3, figsize=(258, 258))
#     for i in range(len(X_batch)):
#         a[0][i].imshow(np.reshape(X_batch[i], (258, 258)))
#         a[1][i].imshow(np.reshape(results[i], (258, 258)))
#
#
#     fig = plt.figure()
#     fig.savefig('full_figure.png')
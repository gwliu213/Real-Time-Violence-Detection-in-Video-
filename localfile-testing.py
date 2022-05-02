#!/usr/bin/env python
# coding: utf-8

# In[8]:


from __future__ import absolute_import
from __future__  import division
from __future__ import print_function
import tensorflow as tf
import numpy as np
from skimage.io import imread
from skimage.transform import resize
import cv2
import numpy as np
import os
from PIL import Image
from io import BytesIO
import time
from visdom import Visdom

# In[9]:


from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())



def mamon_videoFightModel2(tf,wight='mamonbest947oscombo.hdfs'):
    layers = tf.keras.layers
    models = tf.keras.models
    losses = tf.keras.losses
    optimizers = tf.keras.optimizers
    metrics = tf.keras.metrics
    num_classes = 2
    cnn = models.Sequential()
    #cnn.add(base_model)

    input_shapes=(160,160,3)
    np.random.seed(1234)
    vg19 = tf.keras.applications.vgg19.VGG19
    base_model = vg19(include_top=False,weights='imagenet',input_shape=(160, 160,3))
    # Freeze the layers except the last 4 layers
    #for layer in base_model.layers:
    #    layer.trainable = False

    cnn = models.Sequential()
    cnn.add(base_model)
    cnn.add(layers.Flatten())
    model = models.Sequential()

    model.add(layers.TimeDistributed(cnn,  input_shape=(30, 160, 160, 3)))
    model.add(layers.LSTM(30 , return_sequences= True))

    model.add(layers.TimeDistributed(layers.Dense(90)))
    model.add(layers.Dropout(0.1))

    model.add(layers.GlobalAveragePooling1D())

    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.3))

    model.add(layers.Dense(num_classes, activation="sigmoid"))

    adam = optimizers.Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.load_weights(wight)
    rms = optimizers.RMSprop()

    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=["accuracy"])

    return model


# In[11]:


import numpy as np
from skimage.transform import resize
np.random.seed(1234)
model22 = mamon_videoFightModel2(tf)


# In[ ]:


model22._make_predict_function()


# In[ ]:


def video_mamonreader(cv2,filename):
    frames = np.zeros((30, 160, 160, 3), dtype=np.float)
    i=0
    print(frames.shape)
    frame = np.zeros((160,160,3), np.uint8)
    vc = cv2.VideoCapture(filename)
    if vc.isOpened():
        rval , frame = vc.read()
    else:
        rval = False
    frm = resize(frame,(160,160,3))
    frm = np.expand_dims(frm,axis=0)
    if(np.max(frm)>1):
        frm = frm/255.0
    frames[i][:] = frm
    i +=1
    print("reading video")
    while i < 30:
        rval, frame = vc.read()
        frm = resize(frame,(160,160,3))
        frm = np.expand_dims(frm,axis=0)
        if(np.max(frm)>1):
            frm = frm/255.0
        frames[i][:] = frm
        i +=1
    return frames


# In[ ]:


def pred_fight(model,video,acuracy=0.9):
    pred_test = model.predict(video)
    if pred_test[0][1] >=acuracy:
        return True , pred_test[0][1]
    else:
        return False , pred_test[0][1]


def main_fight(vidoss):
    vid = video_mamonreader(cv2,vidoss)
    datav = np.zeros((1, 30, 160, 160, 3), dtype=np.float)
    datav[0][:][:] = vid
    millis = int(round(time.time() * 1000))
    print(millis)
    f , percent = pred_fight(model22,datav,acuracy=0.65)
    millis2 = int(round(time.time() * 1000))
    print(millis2)

    res_mamon = {'fight':f , 'precentegeoffight':str(percent)}
    res_mamon['processing_time'] =  str(millis2-millis)
    print(res_mamon)
    return res_mamon
def visdom_show(vis,img):
    img_vis = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_vis = np.transpose(img_vis, (2, 0, 1))
    vis.image(img_vis, opts={'title': 'SensorDog!', 'caption': 'Click Me!'}, win='Elevator')
    # vis.image(img_vis, opts={'title': 'Anomaly!', 'caption': 'Click Me!'}, win='Anomaly')
    # videoWriter.write(img)
def gaowen_video_reader(vis,filename):
    vc = cv2.VideoCapture(filename)
    video_clip_count=0
    results=[]
    while True:
        frames = np.zeros((30, 160, 160, 3), dtype=np.float)
        i=0
        print(frames.shape)
        frame = np.zeros((160,160,3), np.uint8)
        if vc.isOpened():
            rval , frame = vc.read()
            frm = resize(frame, (160, 160, 3))
            resized=cv2.resize(frame, (800,400), interpolation=cv2.INTER_AREA)
            visdom_show(vis, resized)
        else:
            rval = False
        frm = resize(frame,(160,160,3))
        frm = np.expand_dims(frm,axis=0)
        if(np.max(frm)>1):
            frm = frm/255.0
        frames[i][:] = frm
        i +=1
        print("reading video")
        while i < 30:
            rval, frame = vc.read()
            resized=cv2.resize(frame, (800, 400), interpolation=cv2.INTER_AREA)
            frm = resize(frame,(160,160,3))
            visdom_show(vis, resized)
            frm = np.expand_dims(frm,axis=0)
            if(np.max(frm)>1):
                frm = frm/255.0
            frames[i][:] = frm
            i +=1
        res_mamon = gaowen_update_fight(frames,vis, video_clip_count, i)
        results.append((res_mamon))
        video_clip_count +=1
    return results
def gaowen_update_fight(vid,vis, video_clip_count,i):
    # vid = gaowen_video_reader(cv2,vidoss)
    datav = np.zeros((1, 30, 160, 160, 3), dtype=np.float)
    datav[0][:][:] = vid
    millis = int(round(time.time() * 1000))
    print(millis)
    f , percent = pred_fight(model22,datav,acuracy=0.90)
    millis2 = int(round(time.time() * 1000))
    print(millis2)

    res_mamon = {'frames':str(video_clip_count*30)+str('-')+str(video_clip_count*30+i),'fight':f , 'precentegeoffight':str(percent)}
    if f:
        vis.text('Violence = '+str(percent) +', frames '+str(video_clip_count*30)+str('-')+str(video_clip_count*30+i), win='Evaluation', opts={'title': 'Evaluation!'})
    else:
        vis.text('Violence = '+str(percent) +', frames '+str(video_clip_count*30)+str('-')+str(video_clip_count*30+i), win='Evaluation', opts={'title': 'Evaluation!'})
        # vis.image(img_vis, opts={'title': 'SensorDog!', 'caption': 'Click Me!'}, win='Elevator')
    res_mamon['processing_time'] =  str(millis2-millis)
    print(res_mamon)
    return res_mamon
def gaowen_main(vidoss):
    vis = Visdom(env='Aggression', port='8097')
    assert vis.check_connection()
    vid = gaowen_video_reader(vis, vidoss)
# res = main_fight('/home/gaowen/Documents/MGM/sensordogv2/Sensordog/JOB01_VideoStreamingYolo/video/fall_2.mp4')
# res = gaowen_main('/home/gaowen/Documents/MGM/sensordogv2/Sensordog/JOB01_VideoStreamingYolo/video/home35.mp4')
# res = gaowen_main('/home/gaowen/Documents/MGM/sensordogv2/Sensordog/JOB01_VideoStreamingYolo/video/fall_2.mp4')
res = gaowen_main('/home/gaowen/Documents/MGM/sensordogv2/Sensordog/JOB01_VideoStreamingYolo/video/fight_test.mp4')


# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 14:44:39 2023

@author: User
"""
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
import cv2
import numpy as np

model=tf.keras.models.load_model('C:/Users/User/Downloads/depth_model.h5', compile=False)
model_2=tf.keras.models.load_model('C:/Users/User/Downloads/normal_model.h5', compile=False)

cap = cv2.VideoCapture(0)
while cap.isOpened(): 
    ret, frame = cap.read()
    im=Image.fromarray(frame, 'RGB')
    im = im.resize((640,192))
    img_array = np.array(im)
    img_array = np.expand_dims(img_array, axis=0)
    img_array=img_array.astype(np.float32)
    img_array /= 255.
    prediction = model.predict(img_array).squeeze()
    prediction_normal = model_2.predict(img_array).squeeze()
    
    plt.subplot(2,1,1)
    plt.imshow(prediction, cmap='CMRmap_r')
    plt.xticks([])
    plt.yticks([])
    plt.subplot(2,1,2)
    plt.imshow(prediction_normal)
    plt.xticks([])
    plt.yticks([])
    cv2.imshow('CV2Frame', cv2.resize(frame, (640,192)))
    plt.pause(0.00001)
    if cv2.waitKey(10) & 0xFF == ord('q'): 
        cap.release()
        cv2.destroyAllWindows()
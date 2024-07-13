# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 13:32:24 2023

@author: User
"""

import cv2
import torch
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
import tensorflow.keras.backend as K
import tensorflow_addons as tfa

#opt = tfa.optimizers.AdamW(learning_rate=0.0001, weight_decay=1e-6,amsgrad=True)

def loss_function(y_true, y_pred):

  #Cosine distance loss
  l_depth = K.mean(K.abs(y_pred - y_true), axis=-1)

  # edge loss for sharp edges
  '''
  dy_true, dx_true = tf.image.image_gradients(y_true)
  dy_pred, dx_pred = tf.image.image_gradients(y_pred)
  l_edges = K.mean(K.abs(dy_pred - dy_true) + K.abs(dx_pred - dx_true), axis=-1)
  '''
  # structural similarity loss
  l_ssim = K.clip((1 - tf.image.ssim(y_true, y_pred, 1.0)) * 0.5, 0, 1)

  # weightage
  w1, w3 = 1.0, 0.1
  #+ (w2 * K.mean(l_edges)) w2=1.0
  return (w1 * l_ssim) + (w3 * K.mean(l_depth))
def accuracy_function(y_true, y_pred):
  return K.mean(K.equal(K.round(y_true), K.round(y_pred)))
#midas=tf.keras.models.load_model("C:/Users/User/Downloads/depth_model.h5",custom_objects={'loss_function':loss_function, 'accuracy_function':accuracy_function})
midas=tf.keras.models.load_model("C:/Users/User/Downloads/normal_model.h5")

# Download the MiDaS
midas = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small')
midas.to('cpu')
midas.eval()
# Input transformation pipeline
transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')
transform = transforms.small_transform 

# Hook into OpenCV
cap = cv2.VideoCapture(0)
while cap.isOpened(): 
    ret, frame = cap.read()

    # Transform input for midas 
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    imgbatch = transform(img).to('cpu')

    # Make a prediction
    with torch.no_grad(): 
        prediction = midas(imgbatch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size = img.shape[:2], 
            mode='bicubic', 
            align_corners=False
        ).squeeze()

        output = prediction.cpu().numpy()

        print(output)
    plt.imshow(output)
    cv2.imshow('CV2Frame', frame)
    plt.pause(0.00001)

    if cv2.waitKey(10) & 0xFF == ord('q'): 
        cap.release()
        cv2.destroyAllWindows()

plt.show()

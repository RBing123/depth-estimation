# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 00:34:51 2023

@author: User
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt

pointcloud = np.fromfile("D:/dataset/sequences/00/velodyne/000150.bin",dtype=np.float32).reshape(-1,4)
pointcloud = pointcloud[:,:3].astype(np.float64)
image = cv2.imread("D:/dataset/sequences/00/image_2/000150.png")
a = np.where(pointcloud[:,0]>=0)
pointcloud = pointcloud[a]
plt.rcParams['figure.figsize'] = (120, 35)
plt.imshow(image)
rotation = np.array([7.533745e-03, -9.999714e-01, -6.166020e-04, 1.480249e-02, 7.280733e-04, -9.998902e-01, 9.998621e-01, 7.523790e-03, 1.480755e-02],dtype="float").reshape(3,3)
translation = np.array([-4.069766e-03, -7.631618e-02, -2.717806e-01],dtype="float").reshape(1,3)
distortion = np.array([[0,0,0,0]],dtype="float")
camera = np.array((721.5377,0,609.5593,
        0,721.5377,172.854,
        0,0,1),dtype="float").reshape(3,3)
reTransform = cv2.projectPoints(pointcloud,rotation,translation,camera,distortion)
reTransform = reTransform[0][:,0].astype(int)
pixel = reTransform
filter = np.where((pixel[:,0]<1242)&(pixel[:,1]<375)&(pixel[:,0]>=0)&(pixel[:,1]>=0))
pixel = pixel[filter]
depth = pointcloud[:,0].reshape(-1,1)[filter]
plt.scatter(pixel[:,0], pixel[:,1] , s=100)
plt.imshow(image)
plt.scatter(pixel[:,0], pixel[:,1] ,c=depth, s=100)
plt.imshow(image)
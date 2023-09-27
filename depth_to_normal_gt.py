# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 22:50:08 2023

@author: User
"""
#%%import
import numpy as np
import pandas as pd
import cv2
import os
import sys
import tensorflow as tf
from PIL import Image
from IPython.display import display
from matplotlib import pyplot as plt
#%% import RGB image
path="C:/Users/User/Desktop/vkitti_2.0.3_rgb"
rgb_file=os.listdir(path)
filelist=[]
for subfile in rgb_file:  
    address=path+'/'+subfile
    for sub_subfile in os.listdir(address):
        a=path+'/'+subfile+'/'+sub_subfile+'/'+'frames'+'/'+'rgb'
        for last in os.listdir(a):
            aa=a+'/'+last
            filelist.append(aa)
RGB_image=[]
for dirs in filelist:
    ad=os.listdir(dirs)
    for sec_dirs in ad:
        d=dirs+'/'+sec_dirs
        RGB_image.append(d)
#%% import depth image
path="C:/Users/User/Desktop/vkitti_2.0.3_depth"
depth_file=os.listdir(path)
dfilelist=[]
for subfile in depth_file:  
    address=path+'/'+subfile
    for sub_subfile in os.listdir(address):
        a=path+'/'+subfile+'/'+sub_subfile+'/'+'frames'+'/'+'depth'
        for last in os.listdir(a):
            aa=a+'/'+last
            dfilelist.append(aa)
Depth_image=[]
for dirs in dfilelist:
    ad=os.listdir(dirs)
    for sec_dirs in ad:
        d=dirs+'/'+sec_dirs
        Depth_image.append(d)
#%% depth to surface normal
def depth_to_norm(depth_img, drs, i):
    i=str(i)
    s=i.zfill(5)
    depth_img=cv2.imread(depth_img, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    zy, zx = np.gradient(depth_img)
    normal = np.dstack((-zx, -zy, np.ones_like(depth_img)))
    n = np.linalg.norm(normal, axis=2)
    normal[:, :, 0] /= n
    normal[:, :, 1] /= n
    normal[:, :, 2] /= n
    normal += 1
    normal /= 2
    normal *= 255
    cv2.imwrite(drs+'/'+'normal_'+s+'.png', normal[:, :, ::-1])

path="C:/Users/User/Desktop/vkitti_2.0.3_normal"
normal_file=os.listdir(path)
nfilelist=[]
for subfile in normal_file:
    address=path+'/'+subfile
    for sub_subfile in os.listdir(address):
        a=path+'/'+subfile+'/'+sub_subfile+'/'+'frames'+'/'+'normal'
        for last in os.listdir(a):
            aa=a+'/'+last
            nfilelist.append(aa)
for k in range(20,40):
    for i in range((8940+233*(k-20)),(8940+233*(k-19))):
        depth_to_norm(Depth_image[i],nfilelist[k],(i-8940-233*(k-20)))
#%% import normal image
path="C:/Users/User/Desktop/vkitti_2.0.3_normal"
normal_image=[]
for dirs in nfilelist:
    ad=os.listdir(dirs)
    for sec_dirs in ad:
        d=dirs+'/'+sec_dirs
        normal_image.append(d)
#%% build data
def  transform(img):
    image=cv2.imread(img)
    img=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result=cv2.resize(img, (640,192))
    return result
data={
      "image":[x for x in RGB_image],
      "depth":[x for x in Depth_image],
      "normal":[x for x in normal_image]
      }
df=pd.DataFrame(data)
df=df.sample(frac=1,random_state=42)
'''
for i in range(len(RGB_image)):
    img=transform(RGB_image[i])
    depth_img=cv2.imread(Depth_image[i], cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    depth_img=1/depth_img
    depth_img=cv2.resize(depth_img, (640,192))
    fig, ax=plt.subplots(figsize=(20,20))
    ax.imshow(depth_img)
    plt.imshow(depth_img, cmap='inferno')
'''


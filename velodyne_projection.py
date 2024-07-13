# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 14:28:53 2023

@author: User
"""
#%% import utils
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pandas as pd
import cv2
import os
import sys
import tensorflow as tf
from tensorflow.keras import layers
from torch.utils.data import Dataset
from PIL import Image
from IPython.display import display
from matplotlib import pyplot as plt
#%% pose data
def generate_ground_truth(ground_truth, poses, i):
    ground_truth=np.zeros((len(poses[i]),3,4))
    for j in range(len(poses)):    
        ground_truth[j]=np.array(poses.iloc[j]).reshape(3,4)
    return ground_truth
pose=[]
path = "D:/dataset/poses/"
files = os.listdir(path)
poses=[]
for file in files:
    position=path+file
    pose=pd.read_csv(position, delimiter=' ', header=None)
    poses.append(pose)
    
#%% 檢查poses的維度
for i in range(len(poses)):    
    print(poses[i].shape)
    
#%% 產製地面真值
ground_truth=[]
for i in range(len(poses)):
    ground_truth=generate_ground_truth(ground_truth, poses[i], i)
    #　print(ground_truth.shape)
    
#%% 軌跡圖    
fig=plt.figure(figsize=(7,6))
traj=fig.add_subplot(111,projection='3d')
traj.plot(ground_truth[:,:,3][:,0],ground_truth[:,:,3][:,1], ground_truth[:,:,3][:,2])
traj.set_xlabel('x')
traj.set_ylabel('y')
traj.set_zlabel('z')
      
#%% image data
path ="D:/dataset/sequences/"
image_file = os.listdir(path)
filelist =[]
for file in image_file:
    image_dir = path+file
    filelist.append(image_dir)
im=['image_2', 'image_3', 'velodyne' ,'calib.txt', 'times.txt']
image_pos=[]
for i in range(0,2):
    for f in filelist:
        image_dir=im[i]
        image_absdir=f+'/'+image_dir
        image_pos.append(image_absdir)
image_all_data=[]
for dirs in image_pos:
    add=os.listdir(dirs)
    for sec_dir in add:
        dr=dirs+'/'+sec_dir
        image_all_data.append(dr)
#%% 畫圖
for i in range(len(image_all_data)):
    img=cv2.imread(image_all_data[i])
    fig, ax=plt.subplots(figsize=(6,6))
    ax.imshow(img)
    plt.show()
    
#%% calibration
calibration_path=[]
for i in filelist:
    cal_dir=im[3]
    cal_absdir=i+'/'+cal_dir
    calibration_path.append(cal_absdir)

def read_calib_file(path):
    float_chars = set("0123456789.e+- ")
    data = {}
    with open(path, 'r') as f:
        for line in f.readlines():
            key, value = line.split(':', 1)
            value = value.strip()
            data[key] = value
            if float_chars.issuperset(value):
                # try to cast to float array
                try:
                    data[key] = np.array(list(map(float, value.split(' '))))
                except ValueError:
                    # casting error: data[key] already eq. value, so pass
                    pass

    return data
a2=read_calib_file(calibration_path[0])
# print(a2['P0'].reshape(3,4))
#%% 讀入點雲資料
pointcloud_file=os.listdir(path)
point_position=[]
for pf in filelist:
    point_dir=im[2]
    point_absdir=pf+'/'+point_dir
    point_position.append(point_absdir)
point_all_data=[]
for i in range(len(point_position)):
    add=os.listdir(point_position[i])
    add.sort()
    for r in add:
        point_dirs=point_position[i]+'/'+r
        point_all_data.append(point_dirs)
def load_velodyne_points(filename):
    points = np.fromfile(filename, dtype=np.float32).reshape(-1, 4)
    points[:, 3] = 1.0  # homogeneous
    return points
# a1=load_velodyne_points(point_all_data[0])# example
        
#%% produce depth map
#from __future__ import absolute_import, division, print_function

import os
import numpy as np
from collections import Counter

def load_velodyne_points(filename):
    points = np.fromfile(filename, dtype=np.float32).reshape(-1, 4)
    points[:, 3] = 1.0  # homogeneous
    return points

def read_calib_file(path):
    float_chars = set("0123456789.e+- ")
    data = {}
    with open(path, 'r') as f:
        for line in f.readlines():
            key, value = line.split(':', 1)
            value = value.strip()
            data[key] = value
            if float_chars.issuperset(value):
                # try to cast to float array
                try:
                    data[key] = np.array(list(map(float, value.split(' '))))
                except ValueError:
                    # casting error: data[key] already eq. value, so pass
                    pass

    return data

def sub2ind(matrixSize, rowSub, colSub):
    """Convert row, col matrix subscripts to linear indices
    """
    m, n = matrixSize
    return rowSub * (n-1) + colSub - 1


def generate_depth_map(calib_dir, velo_filename, cam=2, vel_depth=False):
    """Generate a depth map from velodyne data
    """
    # load calibration files
    cam2cam = read_calib_file(os.path.join(calib_dir, 'calib_cam_to_cam.txt'))
    velo2cam = read_calib_file(os.path.join(calib_dir, 'calib_velo_to_cam.txt'))
    velo2cam = np.hstack((velo2cam['R'].reshape(3, 3), velo2cam['T'][..., np.newaxis]))
    velo2cam = np.vstack((velo2cam, np.array([0, 0, 0, 1.0])))

    # get image shape
    im_shape = cam2cam["S_rect_02"][::-1].astype(np.int32)

    # compute projection matrix velodyne->image plane
    R_cam2rect = np.eye(4)
    R_cam2rect[:3, :3] = cam2cam['R_rect_00'].reshape(3, 3)
    P_rect = cam2cam['P_rect_0'+str(cam)].reshape(3, 4)
    P_velo2im = np.dot(np.dot(P_rect, R_cam2rect), velo2cam)

    # load velodyne points and remove all behind image plane (approximation)
    # each row of the velodyne data is forward, left, up, reflectance
    velo = load_velodyne_points(velo_filename)
    velo = velo[velo[:, 0] >= 0, :]

    # project the points to the camera
    velo_pts_im = np.dot(P_velo2im, velo.T).T
    velo_pts_im[:, :2] = velo_pts_im[:, :2] / velo_pts_im[:, 2][..., np.newaxis]

    if vel_depth:
        velo_pts_im[:, 2] = velo[:, 0]

    # check if in bounds
    # use minus 1 to get the exact same value as KITTI matlab code
    velo_pts_im[:, 0] = np.round(velo_pts_im[:, 0]) - 1
    velo_pts_im[:, 1] = np.round(velo_pts_im[:, 1]) - 1
    val_inds = (velo_pts_im[:, 0] >= 0) & (velo_pts_im[:, 1] >= 0)
    val_inds = val_inds & (velo_pts_im[:, 0] < im_shape[1]) & (velo_pts_im[:, 1] < im_shape[0])
    velo_pts_im = velo_pts_im[val_inds, :]

    # project to image
    depth = np.zeros((im_shape[:2]))
    depth[velo_pts_im[:, 1].astype(np.int), velo_pts_im[:, 0].astype(np.int)] = velo_pts_im[:, 2]

    # find the duplicate points and choose the closest depth
    inds = sub2ind(depth.shape, velo_pts_im[:, 1], velo_pts_im[:, 0])
    dupe_inds = [item for item, count in Counter(inds).items() if count > 1]
    for dd in dupe_inds:
        pts = np.where(inds == dd)[0]
        x_loc = int(velo_pts_im[pts[0], 0])
        y_loc = int(velo_pts_im[pts[0], 1])
        depth[y_loc, x_loc] = velo_pts_im[pts, 2].min()
    depth[depth < 0] = 0

    return depth


#%% 點雲畫圖與映射產製深度圖與法向量圖

def target_point(point_all_data, j):
    target=point_all_data[j]
    pointcloud= np.fromfile(target, dtype=np.float32).reshape([-1,4])
    pointcloud=pointcloud[:,:3].astype(np.float64)
    a = np.where(pointcloud[:,0]>=0)
    pointcloud=pointcloud[a]
    plt.rcParams['figure.figsize'] = (120, 35)

# generate depth map
def generate_depth_map(calib,idx,img_dir, gt, velo):
    P2=np.array(read_calib_file(calib[idx])['P2']).reshape(3,4)
    P3=np.array(read_calib_file(calib[idx])['P3']).reshape(3,4)
    #k為固有矩陣r為旋轉矩陣t為平移向量
    k1,r1,t1,_,_,_,_=cv2.decomposeProjectionMatrix(P2)
    print("instrinsic matrix:\n",k1)
    print("rotation matrix:\n", r1)
    print("translation vector:\n", t1.round(4))
    img=cv2.imread(img_dir)
    print("image size:\n", img.shape)
    cx=k1[0,2]
    cy=k1[1,2]
    print('Actual center of image(x,y):',(img.shape[1]/2,img.shape[0]/2))
    print('optical center of image(cx,cy):', (cx,cy))
    extrinsic_mat=np.hstack([r1,t1[:3]])
    origin=np.array([0,0,0,1])
    T=extrinsic_mat.dot(origin).round(4).reshape(-1,1)
    pt=np.array([3,2,4,1]).reshape(-1,1)
    pt_transform=ground_truth[4].dot(pt)
    depth_from_left_cam=pt_transform[2]
    print('point:\n', pt)
    print('Transformed point:\n', pt_transform)
    print('Depth from left camera:\n', depth_from_left_cam)
    px_coordinates =k1.dot(pt_transform)/depth_from_left_cam
    print('Pixel Coordinates on image plane:\n', px_coordinates)
    normalized_pixel_coordinates =np.linalg.inv(k1).dot(px_coordinates)
    print('Normalized pixel coordinates:\n', normalized_pixel_coordinates)
    pt_3D = normalized_pixel_coordinates.T*depth_from_left_cam
    print('restored orginally transformed 3D point:\n', pt_3D.T)
    transform_homogeneous = np.vstack([ground_truth[4], np.array([0,0,0,1])])
    inv_transform_homogeneous=np.linalg.inv(transform_homogeneous)
    pt_3D_homogeneous=np.append(pt_3D, 1)
    restored_pt=inv_transform_homogeneous.dot(pt_3D_homogeneous)
    print('restored orginal homogeneous point:\n', restored_pt)
    
    return depth
i,j=0,0
#a3=generate_depth_map(calibration_path, i, image_all_data, generate_ground_truth(ground_truth,poses,j))
#%%　
import matplotlib.image as mpimg
def sub2ind(matrixSize, rowSub, colSub):
    m, n=matrixSize
    return rowSub*(n-1)+colSub-1
def generate_dense_depth_map(calib_dir, velo_filename, img, i, cam=2, vel_depth=False):
    j=i
    # save image path
    resultImg=os.path.join("D:/dataset/depth/", f'{i}.png')
    # get image shape
    png=mpimg.imread(img)
    IMG_H, IMG_W, _ = png.shape
    # compute projection matrix
    P_rect=np.array(read_calib_file(calib_dir)['P2']).reshape(3,4) 
    R_rect=np.array([[1,0,0],[0,1,0],[0,0,1]])
    Tr_velo_to_cam=np.array(read_calib_file(calib_dir)['Tr']).reshape(3,4)
    data={}
    data['P_rect']=P_rect
    data['T_cam_velo']=Tr_velo_to_cam
    data['T_cam_velo']=np.vstack([data['T_cam_velo'], [0,0,0,1]])
    R_rect = np.insert(R_rect,3,values=[0,0,0],axis=0)
    R_rect = np.insert(R_rect,3,values=[0,0,0,1], axis=1)
    data['T_cam2_velo']=R_rect.dot(data['T_cam_velo'])
    # load velodyne points and remove all behind image plane
    pnt=load_velodyne_points(velo_filename)
    pnt=np.delete(pnt, np.where(pnt[0,:]<0), axis=1)
    proj_lidar=data['P_rect'].dot(data['T_cam2_velo']).dot(np.transpose(pnt))
    cam=np.delete(proj_lidar, np.where(proj_lidar[0,:]<0), axis=1)
    cam[:2,:]/=cam[2,:]
    
    # plot image
    plt.figure(figsize=((IMG_W)/72.0,(IMG_H)/72.0),dpi=72.0, tight_layout=True)
    plt.axis([0,IMG_W,IMG_H,0])
    
    u,v,z = cam
    u_out = np.logical_or(u<0, u>IMG_W)
    v_out = np.logical_or(v<0, v>IMG_H)
    outlier = np.logical_or(u_out, v_out)
    cam = np.delete(cam,np.where(outlier),axis=1)
    u,v,z = cam
    
    plt.scatter([u],[v],c=[z],cmap='inferno_r',alpha=0.5,s=1)
    #plt.title("projection")
    plt.savefig(resultImg,bbox_inches='tight')
    plt.show()
    image_array = np.zeros((IMG_H, IMG_W), dtype=np.int16)
    for i in range(cam.shape[1]):
        x = int(round(u[i]))
        y = int(round(v[i]))
        depth =  int(z[i]*256)
        if 0<x<image_array.shape[1] and 0<y<image_array.shape[0]:
            image_array[y,x] = depth
    image_pil = Image.fromarray(image_array, 'I;16')
    image_pil.save(f'D:/dataset/depth/gt{j}.png')
#13 28464  
for i in range(12,13):
    calib_dir=calibration_path[i]
    for j in range(24122,25183):    
        velo_filename=point_all_data[j]
        img=image_all_data[j]
        generate_dense_depth_map(calib_dir, velo_filename, img, j)

P2=np.array(read_calib_file(calibration_path[0])['P2']).reshape(3,4)
k1,r1,t1,_,_,_,_=cv2.decomposeProjectionMatrix(P2)
R_rect=np.array([[1,0,0],[0,1,0],[0,0,1]])
Tr_velo_to_cam=np.array(read_calib_file(calibration_path[1])['Tr']).reshape(3,4)
velofiles=point_all_data[1]
img=image_all_data[1]
resultImg="D:/dataset/depth/resultDensetest.png"
data={}
data['P_rect_2']=P2
data['T_cam0_velo'] = Tr_velo_to_cam
data['T_cam0_velo'] = np.vstack([data['T_cam0_velo'], [0, 0, 0, 1]])
R_rect = np.insert(R_rect,3,values=[0,0,0],axis=0)
R_rect = np.insert(R_rect,3,values=[0,0,0,1],axis=1)
data['T_cam2_velo'] = R_rect.dot(data['T_cam0_velo'])
# print(data['T_cam2_velo'])
pnt=load_velodyne_points(velofiles)
pnt=np.delete(pnt,np.where(pnt[0,:]<0),axis=1)
proj_lidar=data['P_rect_2'].dot(data['T_cam2_velo']).dot(np.transpose(pnt))
cam=np.delete(proj_lidar,np.where(proj_lidar[2,:]<0),axis=1)
cam[:2,:]/=cam[2,:]
png=mpimg.imread(img)
IMG_H,IMG_W,_ = png.shape
plt.figure(figsize=((IMG_W)/72.0,(IMG_H)/72.0),dpi=72.0, tight_layout=True)
plt.axis([0,IMG_W,IMG_H,0])
#plt.imshow(png)
u,v,z = cam
u_out = np.logical_or(u<0, u>IMG_W)
v_out = np.logical_or(v<0, v>IMG_H)
outlier = np.logical_or(u_out, v_out)
cam = np.delete(cam,np.where(outlier),axis=1)
u,v,z = cam
 
# 将激光投影点绘制到图像平面:绘制激光深度散点图
plt.scatter([u],[v],c=[z],cmap='inferno_r',alpha=0.5,s=1)
plt.title("projection")
plt.savefig(resultImg,bbox_inches='tight')
plt.show()
image_array = np.zeros((IMG_H, IMG_W), dtype=np.int16)
for i in range(cam.shape[1]):
    x = int(round(u[i]))
    y = int(round(v[i]))
    depth =  int(z[i]*256)
    if 0<x<image_array.shape[1] and 0<y<image_array.shape[0]:
        image_array[y,x] = depth
depth=np.zeros((IMG_H,IMG_W))
depth[v.astype(int),u.astype(int)]=z
inds=sub2ind(depth.shape, v, u)
dupe_inds=[item for item, count in Counter(inds).items() if count>1]
for dd in dupe_inds:
    pts=np.where(inds==dd)[0]
    x_loc=int(cam[pts[0], 0])
    y_loc=int(cam[pts[0], 1])
    depth[y_loc, x_loc] = cam[pts, 2].min()
depth[depth<0]=0
image_pil = Image.fromarray(image_array, 'I;16')   
image_pil.save("result_16.png")

data={
      "image" : [x for x in image_all_data if x.endswith(".png")],
      "depth" : [x for x in depth if x.endswith(".png")],
      "normal_surface" : []
      }

# df = pd.DataFrame(data)
# df = df.sample(frac=1, random_state=42)


#%% show image and lidar point
for i in range(103,104):
    velo_point=np.fromfile(point_all_data[i],dtype=np.float32).reshape(-1,4)
    velo_point = velo_point[:,:3].astype(np.float64)
    image = cv2.imread(image_all_data[i])
    a = np.where(velo_point[:,0]>=0)
    velo_point = velo_point[a]
    plt.rcParams['figure.figsize'] = (120, 35)
    plt.imshow(image)
    rotation = np.array([7.533745e-03, -9.999714e-01, -6.166020e-04, 1.480249e-02, 7.280733e-04, -9.998902e-01, 9.998621e-01, 7.523790e-03, 1.480755e-02],dtype="float").reshape(3,3)
    translation = np.array([-4.069766e-03, -7.631618e-02, -2.717806e-01],dtype="float").reshape(1,3)
    distortion = np.array([[0,0,0,0]],dtype="float")
    camera = np.array((721.5377,0,609.5593,
            0,721.5377,172.854,
            0,0,1),dtype="float").reshape(3,3)
    reTransform = cv2.projectPoints(velo_point,rotation,translation,camera,distortion)
    reTransform = reTransform[0][:,0].astype(int)
    pixel = reTransform
    filter = np.where((pixel[:,0]<1242)&(pixel[:,1]<375)&(pixel[:,0]>=0)&(pixel[:,1]>=0))
    pixel = pixel[filter]
    depth = velo_point[:,0].reshape(-1,1)[filter]
    plt.scatter(pixel[:,0], pixel[:,1] , s=100)
    plt.imshow(image)
    plt.scatter(pixel[:,0], pixel[:,1] ,c=depth, s=100)
    plt.imshow(image)

#%% 資料流
class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, data, batch_size=6, dim=(384, 1248), n_channels=3, shuffle=True):
        """
        Initialization
        """
        self.data = data
        self.indices = self.data.index.tolist()
        self.dim = dim
        self.n_channels = n_channels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.min_depth = 0.1
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.data) / self.batch_size))

    def __getitem__(self, index):
        if (index + 1) * self.batch_size > len(self.indices):
            self.batch_size = len(self.indices) - index * self.batch_size
        # Generate one batch of data
        # Generate indices of the batch
        index = self.indices[index * self.batch_size : (index + 1) * self.batch_size]
        # Find list of IDs
        batch = [self.indices[k] for k in index]
        x, y = self.data_generation(batch)

        return x, y

    def on_epoch_end(self):

        """
        Updates indexes after each epoch
        """
        self.index = np.arange(len(self.indices))
        if self.shuffle == True:
            np.random.shuffle(self.index)

    def load(self, image_path, depth_map, mask):
        """Load input and target image."""

        image_ = cv2.imread(image_path)
        image_ = cv2.cvtColor(image_, cv2.COLOR_BGR2RGB)
        image_ = cv2.resize(image_, self.dim)
        image_ = tf.image.convert_image_dtype(image_, tf.float32)

        depth_map = np.load(depth_map).squeeze()

        mask = np.load(mask)
        mask = mask > 0

        max_depth = min(300, np.percentile(depth_map, 99))
        depth_map = np.clip(depth_map, self.min_depth, max_depth)
        depth_map = np.log(depth_map, where=mask)

        depth_map = np.ma.masked_where(~mask, depth_map)

        depth_map = np.clip(depth_map, 0.1, np.log(max_depth))
        depth_map = cv2.resize(depth_map, self.dim)
        depth_map = np.expand_dims(depth_map, axis=2)
        depth_map = tf.image.convert_image_dtype(depth_map, tf.float32)

        return image_, depth_map

    def data_generation(self, batch):

        x = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size, *self.dim, 1))

        for i, batch_id in enumerate(batch):
            x[i,], y[i,] = self.load(
                self.data["image"][batch_id],
                self.data["depth"][batch_id],
                self.data["mask"][batch_id],
            )

        return x, y

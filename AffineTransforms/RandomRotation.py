# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 17:04:26 2021

@author: Purnendu Mishra
"""

import cv2
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt

from skimage.io import imread

#%%
# =============================================================================
# Read the image and labels
# =============================================================================

img_path = Path.cwd()/'test_img.jpg'

image    = imread(img_path)
y_true   = np.load('test_img_labels.npy')

h, w     = image.shape[:2]

y_true[:,0] *= w
y_true[:,1] *= h

xg = y_true[:,0]
yg = y_true[:,1]

# fig = plt.figure(figsize = (7,7))
# plt.imshow(image)
# plt.scatter(xg, yg , c='r')
# plt.show()

#%%
# =============================================================================
# Image Rotation
# =============================================================================

# theta = np.random.uniform(-1,1) * 90 # In degree
# rad   = np.deg2rad(theta)

# # Rotational matrix
# # Rr    = np.array( [[np.cos(rad), np.sin(rad), 0],
# #                    [-np.sin(rad), np.cos(rad), 0]
# #                   ],
# #                  dtype = np.float32
# #                  )

# # Rotating with respect to center of the image
# Rr = cv2.getRotationMatrix2D((w//2, h//2), theta, 1)

# img_rot = cv2.warpAffine(image, Rr, (w,h))

# Rr2 = np.append(Rr, np.array([[0,0,1]]), axis = 0)

# # c   = np.array([[0,0,1]]).T
# # Rr2 = np.append(Rr2, c, axis  = 1)

# y_  = np.append(y_true.T, np.ones((1,21), dtype = np.float32), axis = 0)

# y_rot = np.matmul(Rr2, y_)[:2].T

# xr, yr = y_rot.T


def random_rotation(x, y, max_theta = 30):
    
    theta = np.random.uniform(-1,1) * max_theta
    rad   = np.deg2rad(theta)
    
    scale = np.random.uniform(0.25, 1.0) + 0.25 
    
    h, w  = x.shape[:2]
   
    # Rotating with respect to center of the image
    Rr = cv2.getRotationMatrix2D((w//2, h//2), theta, scale)
    
    # The rotated image will be
    xR = cv2.warpAffine(image, Rr, (w,h))
    
    # The modified Rotation matrix
    Rm = np.append(Rr, np.array([[0,0,1]]), axis = 0)
    
    
    # Rotate the keypoints
    K   = y.shape[0]  # No. of keypoints
    y_  = np.append(y.T, np.ones((1,K), dtype = np.float32), axis = 0)
    
    # The rotated keypoints
    yR  = np.matmul(Rm, y_)[:2].T
    
    for i in range(K):
        a = yR[i, 0]
        b = yR[i, 1]
        
        if a > w - 1 or b > h - 1:
            yR[i] = 0.
        elif a < 0. or b < 0.:
            yR[i] = 0.  
        else:
            pass
    
    # Normalize the keypoints
    yR[:,0] /= w
    yR[:,1] /= h
    
    return (xR, yR)

img_rot, y_rot = random_rotation(x = image, y = y_true, max_theta = 60)

y_rot[:,0] *= w
y_rot[:,1] *= h

xr, yr = y_rot.T

fig = plt.figure(figsize = (10,10))
plt.subplot(1,2,1)
plt.imshow(image)
# plt.scatter(xg, yg, c='r')
plt.subplot(1,2,2)
plt.imshow(img_rot)
plt.scatter(xr, yr, c='r')
plt.show()


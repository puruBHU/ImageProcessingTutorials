# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 13:54:12 2021

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

fig = plt.figure(figsize = (7,7))
plt.imshow(image)
plt.scatter(xg, yg , c='r')
plt.show()

#%%


def random_translation(x, y, alpha = 0.2):
    
    h, w = x.shape[:2]
    
    # The maximum translation along each dimension
    tx   = np.clip(np.random.uniform(-1, 1), -1 * alpha, alpha) * w  # translation along image width
    ty   = np.clip(np.random.uniform(-1, 1), -1 * alpha, alpha) * h  # translation along image height
    
    print(tx, ty)
    
    # The translation matrix
    T  = np.array([[1,0,tx], [0,1,ty]], dtype = np.float32)
    
    # The image translation
    xT = cv2.warpAffine(x, T, (w, h))
    
    # Translation of hand keypoints
    ## Converting into homogenous coordinate system
    T_ = np.append(T , np.array([[0,0,1]], dtype = np.float32), axis = 0)
    
    K  = y.shape[0]
    # Converting keypoints coordinates into homogenous coordinate system
    y_ = np.append(y.T, np.ones((1,K), dtype = np.float32), axis = 0)  # shape = 3 X K
    yT = np.matmul(T_, y_)[:2,:].T
    
    # Limit the keypoints between 0 and h or w
    # yT[:,0] = yT[:,0].clip(min = 0, max = w - 1) / w
    # yT[:,1] = yT[:,1].clip(min = 0, max = h - 1) / h
    
    for i in range(K):
        a = yT[i, 0]
        b = yT[i, 1]
        
        if a > w - 1 or b > h - 1:
            yT[i] = 0.
        elif a < 0. or b < 0.:
            yT[i] = 0.  
        else:
            pass
        
        
    yT[:,0] /= w
    yT[:,1] /= h
    
    return (xT, yT)
    

translated_img, y_cap = random_translation(x = image, y = y_true, alpha = 0.2)

xg2 = y_cap[:,0] * 256
yg2 = y_cap[:,1] * 256

fig = plt.figure(figsize = (10,10))
plt.subplot(1,2,1)
plt.imshow(image)
plt.scatter(xg, yg, c='r')
plt.subplot(1,2,2)
plt.imshow(translated_img)
plt.scatter(xg2, yg2, c='r')
plt.show()

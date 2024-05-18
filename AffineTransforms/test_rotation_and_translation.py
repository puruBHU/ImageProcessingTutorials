# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 18:27:21 2021

@author: Purnendu Mishra
"""

import cv2
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt

from skimage.io import imread


#%%
def random_translation(x, y, alpha = 0.2):
    
    h, w = x.shape[:2]
    
    y[:,0] *= w
    y[:,1] *= h
    
    # The maximum translation along each dimension
    tx   = np.random.uniform(-alpha, alpha) * w  # translation along image width
    ty   = np.random.uniform(-alpha, alpha) * h  # translation along image height
    
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


def random_rotation(x, y, max_theta = 30):
    
    theta = np.random.uniform(-1.,1.) * max_theta # Angle in degrees
    
    scale = np.random.uniform(0.25, 1.0) + 0.25 
    
    h, w  = x.shape[:2]
    
    y[:,0] *= w
    y[:,1] *= h
   
    # Rotating with respect to center of the image
    Rr = cv2.getRotationMatrix2D((w//2, h//2), theta, scale)
    
    # The rotated image will be
    xR = cv2.warpAffine(x, Rr, (w,h))
    
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

#%%
if __name__=='__main__':
    
    
    img_path = Path.cwd()/'test_img.jpg'

    x        = imread(img_path)
    image    = x.copy()
    y        = np.load('test_img_labels.npy')

    h, w     = x.shape[:2]
    
    if np.random.rand() > 0.5:
        print('random translation')
        x, y       = random_translation(x = x, y = y, alpha = 0.2)
        
    if np.random.rand() > 0.5:    
        print('random rotation')
        x, y       = random_rotation(x = x, y = y, max_theta = 45)
    
    y[:,0] *= w
    y[:,1] *= h
    
    px , py = y.T
    
    fig = plt.figure(figsize = (10,10))
    plt.subplot(1,2,1)
    plt.imshow(image)
    plt.subplot(1,2,2)
    plt.imshow(x)
    plt.scatter(px, py, c='r')
    plt.show()

 
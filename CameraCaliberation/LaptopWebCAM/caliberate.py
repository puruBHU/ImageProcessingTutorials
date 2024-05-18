# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 11:20:28 2021

@author: Purnendu Mishra
"""

import cv2
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# prepare object points
nx = 8 #number of inside corners in x
ny = 6# number of inside corners in y
# Make a list of calibration images
fname = 'CheckBoardImage/checkerboard_12.jpg'
img   = cv2.imread(fname)
# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)



# Find the chessboard corners
ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
# If found, draw corners
if ret == True:
    # Draw and display the corners
    cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
    fig = plt.figure(figsize = (10,10))
    plt.imshow(img)
    plt.show()



# plt.imshow(gray)
# plt.show()
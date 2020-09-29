#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 19:55:30 2020

@author: jake
"""


import cv2 as cv
import numpy as np
import time
from matplotlib import pyplot as plt

img = cv.imread('./env_pics/0_3_270_1.jpg', 0)
# cv.imshow("input image", img)
kernel = np.array([[0, -1, 0],
                   [-1, 7, -1],
                   [0, -1, 0]], np.float32)
dst = cv.filter2D(img, -1, kernel=kernel)
edges = cv.Canny(dst,100,200)
adp = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,13,2)
edges1 = cv.Canny(adp,100,200)
cv.imshow("custom_blur_demo", dst)
cv.imshow("edges", edges)
cv.imshow("edges1", edges1)
cv.imshow('adp',adp)
cv.waitKey(10000)
cv.destroyAllWindows()

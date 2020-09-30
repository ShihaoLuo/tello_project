#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 14:10:40 2020

@author: jake
"""


import cv2 as cv


def positive_image_create(img):
    cv.namedWindow('roi', cv.WINDOW_NORMAL)
    cv.imshow('roi', img)
    roi = cv.selectROI('roi', img, True, False)
    cv.destroyAllWindows()
    x, y, w, h = roi
    cv.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
    dst = img[y:y+h, x:x+w]
    while True:
        cv.imshow('dst', dst)
        a = cv.waitKey(10)
        if a == ord('q'):
            cv.destroyAllWindows()
            break


img = cv.imread('../env_pics/0_3_270_1.jpg')
positive_image_create(img)


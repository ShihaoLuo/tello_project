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

def draw_approx_hull_polygon(img, cnts):
    # img = np.copy(img)
    img = np.zeros(img.shape, dtype=np.uint8)

    cv.drawContours(img, cnts, -1, (255, 0, 0), 2)  # blue

    '''epsilion = img.shape[0]/32
    approxes = [cv.approxPolyDP(cnt, epsilion, True) for cnt in cnts]
    cv.polylines(img, approxes, True, (0, 255, 0), 2)  # green

    hulls = [cv.convexHull(cnt) for cnt in cnts]
    cv.polylines(img, hulls, True, (0, 0, 255), 2)  # red'''

    # 我个人比较喜欢用上面的列表解析，我不喜欢用for循环，看不惯的，就注释上面的代码，启用下面的
    # for cnt in cnts:
    #     cv2.drawContours(img, [cnt, ], -1, (255, 0, 0), 2)  # blue
    #
    #     epsilon = 0.01 * cv2.arcLength(cnt, True)
    #     approx = cv2.approxPolyDP(cnt, epsilon, True)
    #     cv2.polylines(img, [approx, ], True, (0, 255, 0), 2)  # green
    #
    #     hull = cv2.convexHull(cnt)
    #     cv2.polylines(img, [hull, ], True, (0, 0, 255), 2)  # red
    return img

def draw_min_rect_circle(img, cnts):  # conts = contours
    img = np.copy(img)

    for cnt in cnts:
        x, y, w, h = cv.boundingRect(cnt)
        cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)  # blue

        '''min_rect = cv.minAreaRect(cnt)  # min_area_rectangle
        min_rect = np.int0(cv.boxPoints(min_rect))
        cv.drawContours(img, [min_rect], 0, (0, 255, 0), 2)  # green

        (x, y), radius = cv.minEnclosingCircle(cnt)
        center, radius = (int(x), int(y)), int(radius)  # for the minimum enclosing circle
        img = cv.circle(img, center, radius, (0, 0, 255), 2)  # red'''
    return img

'''
img = cv.imread('./env_pics/0_3_270_1.jpg', 0)
kernel = np.array([[0, -1, 0],
                   [-1, 6, -1],
                   [0, -1, 0]], np.float32)
dst = cv.filter2D(img.copy(), -1, kernel=kernel)
edges = cv.Canny(dst,100,200)
thresh, contours, hierarchy = cv.findContours(edges, 
                                               cv.RETR_EXTERNAL, 
                                               cv.CHAIN_APPROX_SIMPLE)

img1 = draw_min_rect_circle(img, contours)
cv.imshow('img',img)
cv.imshow("custom_blur_demo", dst)
cv.imshow("edges", edges)
cv.imshow('contours',img1)
cv.waitKey(100000)
cv.destroyAllWindows()
'''

def run():
    image = cv.imread('./env_pics/0_3_270_1.jpg')  # a black objects on white image is better

    # gray = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
    # ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    thresh = cv.Canny(image, 128, 256)

    thresh, contours, hierarchy = cv.findContours(thresh, cv.RETR_EXTERNAL, 
                                                   cv.CHAIN_APPROX_SIMPLE)
    # print(hierarchy, ":hierarchy")
    """
    [[[-1 -1 -1 -1]]] :hierarchy  # cv2.Canny()
    
    [[[ 1 -1 -1 -1]
      [ 2  0 -1 -1]
      [ 3  1 -1 -1]
      [-1  2 -1 -1]]] :hierarchy  # cv2.threshold()
    """

    imgs = [
        image, thresh,
        draw_min_rect_circle(image, contours),
        draw_approx_hull_polygon(image, contours),
    ]

    for img in imgs:
        # cv2.imwrite("%s.jpg" % id(img), img)
        cv.imshow("contours", img)
        cv.waitKey(1943)


if __name__ == '__main__':
    run()
pass

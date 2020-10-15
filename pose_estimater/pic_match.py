#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 11:04:13 2020

@author: jake
"""

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import json
import multiprocessing
#import set_world_point

object_name = 'showcan'

def save_2_jason(_file, arr):
    data = {}
    cnt = 0
    for i in arr:
        data['KeyPoint_%d' % cnt] = []
        data['KeyPoint_%d' % cnt].append({'x': i.pt[0]})
        data['KeyPoint_%d' % cnt].append({'y': i.pt[1]})
        data['KeyPoint_%d' % cnt].append({'size': i.size})
        cnt += 1
    with open(_file, 'w') as outfile:
        json.dump(data, outfile)


def save_2_npy(_file, arr):
    np.save(_file, arr)


def read_from_jason(_file):
    result = []
    with open(_file) as json_file:
        data = json.load(json_file)
        cnt = 0
        while(data.__contains__('KeyPoint_%d' % cnt)):
            pt = cv.KeyPoint(x=data['KeyPoint_%d' % cnt][0]['x'],
                             y=data['KeyPoint_%d' % cnt][1]['y'],
                             _size=data['KeyPoint_%d' % cnt][2]['size'])
            result.append(pt)
            cnt += 1
    return result


def read_from_npy(_file):
    return np.load(_file)


def get_ROI(_img):
    roi = cv.selectROI('roi', _img, True, False)
    cv.destroyAllWindows()
    x, y, w, h = roi
    cv.rectangle(_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
    dst = _img[y:y + h, x:x + w]
    return dst


MIN_MATH_COUNT = 20

img_test = cv.imread('./dataset/'+object_name+'/images/'+object_name+'28.jpg', 0)
img_query = cv.imread('./dataset/'+object_name+'/images/'+object_name+'11.jpg', 0)
img_query = get_ROI(img_query)
img_test = get_ROI(img_test)
sift_paras = dict(nfeatures=0,
                 nOctaveLayers=3,
                 contrastThreshold=0.05,
                 edgeThreshold=10,
                 sigma=0.8)
cv.imwrite('./dataset/'+object_name+'/images/'+object_name+'.jpg',img_query)
'''surf_paras = dict(hessianThreshold=100,
                  nOctaves=10,
                  nOctaveLayers=2,
                  extended=1,
                  upright=0)
surf = cv.xfeatures2d.SURF_create(**surf_paras)'''
sift = cv.xfeatures2d.SIFT_create(**sift_paras)
kp_query, des_query = sift.detectAndCompute(img_query, None)
save_2_jason('dataset/'+object_name+'/kp.json',kp_query)
save_2_npy('dataset/'+object_name+'/des.npy',des_query)

#save_2_jason('kp_query.jason', kp_query)
#save_2_npy('des_query.npy', des_query)
kp_test, des_test = sift.detectAndCompute(img_test, None)
#kp_query_1 = read_from_jason('kp_goodm.jason')
#des_query_1 = read_from_npy('des_goodm.npy')
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)

flann = cv.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(des_query, des_test, k=2)

good = []
kp_good_match_query = []
des_good_match_query = []
for m, n in matches:
    if m.distance < 0.9*n.distance:
        good.append(m)
        print('--------------------\n')
        print('m.imgIdx: {}\n'.format(m.imgIdx))
        print('m.queryIdx: {}\n'.format(m.queryIdx))
        print('m.trainIdx: {}\n'.format(m.trainIdx))
        print('kp_query: {}\n'.format(kp_query[m.queryIdx].pt))
        print('kp_test: {}\n'.format(kp_test[m.trainIdx].pt))
        kp_good_match_query.append(kp_query[m.queryIdx])
        des_good_match_query.append(des_query[m.queryIdx])
print('the num of finding featurs of query is {}\n'.format(len(des_query)))
print('the num of finding featurs of test is {}\n'.format(len(des_test)))
print('the num of finding matches is {}\n'.format(len(matches)))
print("the len of good match is {}\n".format(len(good)))
#save_2_jason('dataset/'+object_name+'/kp.json',kp_good_match_query)
#save_2_npy('dataset/'+object_name+'/des.npy',des_good_match_query)
if len(good)>=MIN_MATH_COUNT:
    src_pts = np.float32([kp_good_match_query[i].pt for i in range(len(kp_good_match_query))]).reshape(-1,1,2)
    #src_pts = np.float32([kp_query_1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    dst_pts = np.float32([kp_test[m.trainIdx].pt for m in good]).reshape(-1,1,2)

    M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()
    h,w = img_query.shape
    d = 1
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv.perspectiveTransform(pts,M)
    img_test = cv.polylines(img_test,[np.int32(dst)],True,255,1,cv.LINE_AA)

else:
    print("Not enough matchs are found - {}/{}".format(len(good),MIN_MATH_COUNT))
    matchesMask = None
draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = None,
                   matchesMask = matchesMask,
                   flags = 2)
img = cv.drawMatches(img_query,kp_query,img_test,kp_test,good,None,**draw_params)

def show_pic(_img):
    fig = plt.figure(figsize=(12, 10))
    plt.subplot(1, 1, 1).axis("off")
    plt.imshow(_img)
    plt.show()

thread = multiprocessing.Process(target=show_pic, args=(img,))
thread.start()
wpixel = np.array([])
wpoint = np.array([])
for i in range(4):
    wpxlx = input('input the x of No.{} wpixel:'.format(i+1))
    wpxly = input('input the y of No.{} wpixel:'.format(i+1))
    wptx = input('input the x of No.{} wpoint:'.format(i+1))
    wpty = input('input the y of No.{} wpoint:'.format(i+1))
    wptz = input('input the z of No.{} wpoint:'.format(i + 1))
    wpxl = np.array([wpxlx, wpxly])
    wpt = np.array([wptx, wpty, wptz])
    wpixel =np.append(wpixel, wpxl)
    wpoint = np.append(wpoint, wpt)
save_2_npy('dataset/'+object_name+'/wpixel.npy', wpixel)
save_2_npy('dataset/'+object_name+'/wpoint.npy', wpoint)
print(wpixel)
print(wpoint)

'''
img_gkp = cv.drawKeypoints(img_query, kp_good_match_query, None)
plt.imshow(img_gkp)
wps = [[None, None, None]]*len(kp_good_match_query)
# pixelpts = [[541.0, 318.5], [625.0, 268.5], [445.0, 301.0], [433.0, 315.6], [413.0, 338.0], [450.0, 440.0], [479.0, 335.5], [482.0, 314.0], [486.0, 305.0]]
# worldpts = [[-35.0,81.0, 22.0], [-73.0, 60.0, 21.0], [9.0, 67.0, 41.4], [14.2, 76.9, 41.4], [23.0, 86.5,41.4],[-5.0, 88.0, 41.4],[-12.5, 85.5, 41.4],[-14.0, 77.0, 41.4],[-15.5, 72.0, 41.4]]
#for i in range(len(pixelpts)):
  #  set_world_point.setworldpoint(wps, kp_good_match_query, pixelpts[i], 5, worldpts[i])

img = cv.drawKeypoints(img_query, kp_query, None)
img2 = cv.drawKeypoints(img_test, kp_test, None)
fig = plt.figure(figsize=(22, 10))
plt.subplot(1, 2, 1)
plt.subplot(1, 2, 1).axis("off")
plt.imshow(img)
plt.subplot(1, 2, 2)
plt.subplot(1, 2, 2).axis("off")
plt.imshow(img2)
plt.show()
222.9 , 115.6 ,  77.95, 117.  ,  78.6 , 257.2 , 218.8 , 255.8 
'''

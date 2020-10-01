#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 16:11:59 2020

@author: jake
"""

import os
import cv2
import time

def process_negpic(_folder):
    file_list = os.listdir(_folder)
    a = 0
    f = open('bg.txt', 'w')
    for i in file_list:
        used_name = _folder + i
        new_name = _folder + 'negative' + str(a) + '.jpg'
        f.write(new_name)
        f.write('\n')
        a += 1
        os.rename(used_name, new_name)
        print("picture {} rename to {}".format(used_name, new_name))
    f.close()


def select_roi(img):
    cv2.namedWindow('roi')
    text = ' use mouse to select the roi'
    cv2.putText(img, text, (img.shape[0]/15, img.shape[1]/15),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (55, 255, 155), 2)
    roi = cv2.selectROI('roi', img)
    x, y, w, h = roi
    dst = img.copy()[y:y+h, x:x+w]
    cv2.imwrite('./positive/positive0.jpg', dst)
    cv2.waitKey(5000)
    cv2.destroyWindow('roi')
    return roi


'''def process_pospic(_folder):
    file_list = os.listdir(_folder)
    a = 1
    f = open('info.dat', 'w')
    for i in file_list:
        used_name = _folder + i
        print
        img = cv2.imread(used_name, 0)
        roi = select_roi(img)
        new_name = _folder + 'positive' + str(a) + '.jpg'
        os.rename(used_name, new_name)
        f.write('{} 1 {} {} {} {}'.format(new_name, roi[0], roi[1], roi[2], roi[3]))
        f.write('\n')
        a += 1
        print("picture {} rename to {}".format(used_name, new_name))
    f.close()'''


def create_possample():
    os.system('opencv_createsamples -show -img positive/positive0.jpg -num 1500 -vec positives.vec -bg bg.txt -pngoutput info')

def train():
    if not os.path.exists('data'):
        os.system('mkdir data')
    os.system('opencv_traincascade -data data -vec positives.vec -bg bg.txt -numPos 1000 -numNeg 500  -numStages 15')

#process_negpic('negative/')
#img = cv2.imread('../env_pics/0_3_270_1.jpg', 0)
#select_roi(img)
#create_possample()
train()
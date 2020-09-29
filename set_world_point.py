#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 13:24:34 2020

@author: jake
"""
import numpy as np


def setworldpoint(wps, kps, pixelpoint, offsetpixel, worldpoint):
    keypoint = np.array([0, 0])
    for i in range(len(kps)):
        keypoint[0] = kps[i].pt[0]
        keypoint[1] = kps[i].pt[1]
        offsetx = abs(keypoint[0] - pixelpoint[0])
        offsety = abs(keypoint[1] - pixelpoint[1])
        if offsetx < offsetpixel and offsety < offsetpixel:
            wps[i]=worldpoint
            break
            
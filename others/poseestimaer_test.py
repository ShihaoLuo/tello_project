# -*- coding: utf-8 -*-
# @Time    : 2021/2/3 下午4:03
# @Author  : JakeShihao Luo
# @Email   : jakeshihaoluo@gmail.com
# @File    : poseestimaer_test.py.py
# @Software: PyCharm

from pose_estimater.pose_estimater import *
import cv2 as cv

p = PoseEstimater(min_match=20)
p.loaddata('pose_estimater/dataset/')
img_test = cv.imread("pose_estimater/dataset/table1/images/table_test5.jpg")
p.estimate_pose(img_test)
import pose_estimater
import numpy as np
import cv2 as cv

obj = pose_estimater.PoseEstimater()
obj.loaddata('dataset/')
obj.showdataset('post')
obj.show_match_start()
img_query = cv.imread('dataset/post/images/post.jpg', 0)
img_test = cv.imread('dataset/post/images/post3.jpg', 0)
pose = obj.estimate_pose(img_query, img_test)
#pose = obj.estimate_pose(img_query, img_test)
#obj.show_transedpxel(img_test)
print('pose:\n{}'.format(pose))
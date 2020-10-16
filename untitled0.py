from pose_estimater import pose_estimater
import cv2 as cv


img = cv.imread('pose_estimater/dataset/toolholder/images/toolholder30.jpg', 0)
img_query = cv.imread('pose_estimater/dataset/toolholder/images/toolholder.jpg', 0)
obj = pose_estimater.PoseEstimater('SIFT', 15)
obj.loaddata('pose_estimater/dataset/toolholder')
pose = obj.estimate_pose(img_query, img, 1)
print(pose)
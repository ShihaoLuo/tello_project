from demon.pose_estimater import Demon
import cv2 as cv
import time

demon = Demon()
img1 = cv.imread('/home/jake/tello_project/pose_estimater/dataset/post/images/post1.jpg')
img2 = cv.imread('/home/jake/tello_project/pose_estimater/dataset/post/images/post5.jpg')

for i in range(5):
    now  = time.time()
    if i % 2 == 0:
        rot, tvec = demon.estimatepose(img1, img2)
    else:
        rot, tvec = demon.estimatepose(img2, img1)
    print('time used: {}'.format(time.time() - now))
    print(rot)
    print(tvec)
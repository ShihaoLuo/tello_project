# -*- coding: utf-8 -*-
# @Time    : 2021/2/9 下午5:20
# @Author  : JakeShihao Luo
# @Email   : jakeshihaoluo@gmail.com
# @File    : face_test.py
# @Software: PyCharm
import face_recognition
import cv2 as cv
import matplotlib.pyplot as plt

# image = cv.imread('./249.0-284.0-228.0-359.0.jpg')
# image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
# cv.imwrite('./11111.jpg', image)
image = face_recognition.load_image_file('./192.168.1.104-up-camera_screenshot_22.02.2021.png')
face_locations = face_recognition.face_locations(image,number_of_times_to_upsample=2)
print(face_locations)
plt.imshow(image)
plt.show()

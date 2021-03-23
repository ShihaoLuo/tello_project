import cv2 as cv
import os

files = os.listdir('./target_images')
for t in files:
    img = cv.imread('./target_images/' + t)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    cv.imwrite('./target_images/' + t, img)

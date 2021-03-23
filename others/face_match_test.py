import cv2 as cv
import face_recognition
import time
import numpy as np

def get_ROI(_img):
    roi = cv.selectROI('roi', _img, True, False)
    cv.destroyAllWindows()
    x, y, w, h = roi
    dst = _img[y:y + h, x:x + w]
    return dst

name = 'luoshihao'
img_query = cv.imread('./target_images/'+name+'/'+name+'2.jpg')
img_query = get_ROI(img_query)
cv.imwrite('./target_images/' + name + '/' + name + '.jpg', img_query)
known_image = face_recognition.face_encodings(img_query)[0]
img_test = cv.imread('./target_images/'+name+'/'+name+'9.jpg')
old = time.time()
locate = face_recognition.face_locations(img_test)
unknown_image = face_recognition.face_encodings(img_test, locate)[0]
result = face_recognition.compare_faces([known_image], unknown_image)
np.save('./target_images/' + name + '/' + name, known_image)
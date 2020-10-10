import numpy as np
import cv2 as cv

pt1 = np.array([37.8, 70.16, 1])
pt2 = np.array([257.13, 70.688, 1])
pt3 = np.array([248.8, 279.6, 1])
pt4 = np.array([39.6, 281.18, 1])

M = np.array([[ 8.78771481e-01, -1.74986412e-03,  2.17804793e+02],
 [ 1.06895080e-02,  8.55676506e-01,  2.20940648e+02],
 [-3.60026515e-06, -6.50585684e-06,  1.00000000e+00]])

npt1 = np.dot(M, pt1)
npt2 = np.dot(M ,pt2)
npt3 = np.dot(M ,pt3)
npt4 = np.dot(M ,pt4)
print(' {} \n {} \n {} \n {}'.format(npt1,npt2,npt3,npt4))

pixelpts = [[250.89958452, 281.37897506],[443.63960951, 284.17530205],[252.11211685, 461.96307247],[435.95387546, 462.84734867]]
worldpts = [[68.5, 0.0, 0.0], [0.0, 0.0, 0.0], [68.5, 68.5, 0.0], [0.0, 68.5, 0.0]]
#pixelpts1 = [[90.9, 244.9], [215.56, 243.5], [219.1, 314.1], [95.75, 314.8]]
#worldpts1 = [[54.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 30.0, 0.0], [54.0, 30.0, 0.0]]
point_world = np.array(worldpts)
point_pixel = np.array(pixelpts)
camera_matrix = np.load('dataset/camera_matrix_tello.npy')
distor_matrix = np.load('dataset/distor_matrix_tello.npy')
pnppara = dict(objectPoints=point_world,
               imagePoints=point_pixel,
               cameraMatrix=camera_matrix,
               distCoeffs=distor_matrix,
               useExtrinsicGuess=0 ,
               flags=cv.SOLVEPNP_ITERATIVE)
_, rvec, tvec= cv.solvePnP(**pnppara)
rotM = np.array(cv.Rodrigues(rvec)[0])
#pose = -rotM.I*np.array(tvec)
pose = np.dot(-rotM.T, tvec)
print(rvec)
print(rotM)
print(pose)
print('----------\n')
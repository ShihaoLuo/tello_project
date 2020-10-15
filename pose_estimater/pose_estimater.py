import numpy as np
import cv2 as cv
import json
import os
import matplotlib.pyplot as plt
import multiprocessing

class PoseEstimater():
    def __init__(self, _algorithm, min_match):
        self.camera_matrix = None
        self.distor_matrix = None
        self.min_match = min_match
        self.algorithm = _algorithm
        self.dataset = {}
        if _algorithm == 'SURF':
            self.detecter = cv.xfeatures2d.SURF_create(hessianThreshold=100, nOctaves=10, nOctaveLayers=2, extended=1, upright=0)
        elif _algorithm == 'SIFT':
            self.detecter = cv.xfeatures2d.SIFT_create(nfeatures=0, nOctaveLayers=3, contrastThreshold=0.05, edgeThreshold=10, sigma=0.8)
        self.queue = multiprocessing.Queue()
        self.show_match = multiprocessing.Process(target=self.show_match)
        self.show_match.start()

    def loaddata(self, _dataset_path):
        self.camera_matrix = np.load(_dataset_path+'camera_matrix_tello.npy')
        self.distor_matrix = np.load(_dataset_path+'distor_matrix_tello.npy')
        listdir = os.listdir(_dataset_path)
        for dir in listdir:
            if os.path.isdir(_dataset_path+dir):
                des = self.read_from_npy(_dataset_path+dir+'/des.npy')
                kp = self.read_from_jason(_dataset_path+dir+'/kp.json')
                wpxl = self.read_from_npy(_dataset_path+dir+'/wpixel.npy')
                wpt = self.read_from_npy(_dataset_path+dir+'/wpoint.npy')
                _dataset = dict()
                _dataset['des'] = des
                _dataset['kp'] = kp
                _dataset['wpixel'] = wpxl
                _dataset['wpoint'] = wpt
                self.dataset[dir] = _dataset

    def read_from_jason(self, _file):
        result = []
        with open(_file) as json_file:
            data = json.load(json_file)
            cnt = 0
            while (data.__contains__('KeyPoint_%d' % cnt)):
                pt = cv.KeyPoint(x=data['KeyPoint_%d' % cnt][0]['x'],
                                 y=data['KeyPoint_%d' % cnt][1]['y'],
                                 _size=data['KeyPoint_%d' % cnt][2]['size'])
                result.append(pt)
                cnt += 1
        return result

    def save_2_jason(self, _file, arr):
        data = {}
        cnt = 0
        for i in arr:
            data['KeyPoint_%d' % cnt] = []
            data['KeyPoint_%d' % cnt].append({'x': i.pt[0]})
            data['KeyPoint_%d' % cnt].append({'y': i.pt[1]})
            data['KeyPoint_%d' % cnt].append({'size': i.size})
            cnt += 1
        with open(_file, 'w') as outfile:
            json.dump(data, outfile)

    def save_2_npy(self, _file, arr):
        np.save(_file, arr)

    def read_from_jason(self, _file):
        result = []
        with open(_file) as json_file:
            data = json.load(json_file)
            cnt = 0
            while (data.__contains__('KeyPoint_%d' % cnt)):
                pt = cv.KeyPoint(x=data['KeyPoint_%d' % cnt][0]['x'],
                                 y=data['KeyPoint_%d' % cnt][1]['y'],
                                 _size=data['KeyPoint_%d' % cnt][2]['size'])
                result.append(pt)
                cnt += 1
        return result

    def read_from_npy(self, _file):
        return np.load(_file)

    def pic_match(self, _img_query, _img, flag):
        img_test = _img
        img_query = _img_query
        kp_test, des_test = self.detecter.detectAndCompute(img_test, None)
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv.FlannBasedMatcher(index_params, search_params)
        for obj in self.dataset.keys():
            des_query = self.dataset[obj]['des']
            kp_query = self.dataset[obj]['kp']
            matches = flann.knnMatch(des_query, des_test, k=2)
            good = []
            kp_good_match_query = []
            des_good_match_query = []
            for m, n in matches:
                if m.distance < 0.56 * n.distance:
                    good.append(m)
                    '''print('--------------------\n')
                    print('m.imgIdx: {}\n'.format(m.imgIdx))
                    print('m.queryIdx: {}\n'.format(m.queryIdx))
                    print('m.trainIdx: {}\n'.format(m.trainIdx))
                    print('kp_query: {}\n'.format(kp_query[m.queryIdx].pt))
                    print('kp_test: {}\n'.format(kp_test[m.trainIdx].pt))'''
                    kp_good_match_query.append(kp_query[m.queryIdx])
                    des_good_match_query.append(des_query[m.queryIdx])
            '''print('the num of finding featurs of query is {}\n'.format(len(des_query)))
            print('the num of finding featurs of test is {}\n'.format(len(des_test)))
            print('the num of finding matches is {}\n'.format(len(matches)))'''
            print("good mathch: {}".format(len(good)))
            if len(good) > self.min_match:
                src_pts = np.float32([kp_good_match_query[i].pt for i in range(len(kp_good_match_query))]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp_test[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
                if flag == 1 and M is not None and mask is not None:
                    h, w = img_query.shape
                    matchesMask = mask.ravel().tolist()
                    d = 1
                    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
                    dst = cv.perspectiveTransform(pts, M)
                    img_test = cv.polylines(img_test, [np.int32(dst)], True, 255, 1, cv.LINE_AA)
                    draw_params = dict(matchColor=(0, 255, 0),
                                       singlePointColor=None,
                                       matchesMask=matchesMask,
                                       flags=2)
                    mimg = cv.drawMatches(img_query, kp_query, img_test, kp_test, good, None, **draw_params)
                    self.queue.put(mimg)
                return obj, M
        return None, None

    def estimate_pose(self, _img_query, _img, flag):
        obj, M = self.pic_match(_img_query, _img, flag)
        if obj is not None and M is not None:
            wpxel = self.dataset[obj]['wpixel'].reshape(-1,2)
            wpxel = np.insert(wpxel, 2, 1, axis=1)
            wpt = self.dataset[obj]['wpoint'].reshape(-1,3)
            wpxel = np.dot(M, wpxel.T).T
            wpxel = np.delete(wpxel, 2, axis=1)
            pnppara = dict(objectPoints=wpt,
                           imagePoints=wpxel,
                           cameraMatrix=self.camera_matrix,
                           distCoeffs=self.distor_matrix,
                           useExtrinsicGuess=0,
                           flags=cv.SOLVEPNP_ITERATIVE)
            _, rvec, tvec = cv.solvePnP(**pnppara)
            rotM = np.array(cv.Rodrigues(rvec)[0])
            pose = np.dot(-rotM.T, tvec)
            pose = pose/2
            return pose
        else:
            return None

    def show_pic(self, _img):
        fig = plt.figure(figsize=(12, 10))
        plt.subplot(1, 1, 1).axis("off")
        plt.imshow(_img)
        plt.show()

    def showdataset(self):
        for i in self.dataset.keys():
            print(self.dataset[i])

    def show_match(self):
        print('in the show thread.')
        while True:
            if not self.queue.empty():
                f = self.queue.get()
                cv.imshow('match_img', f)
            a = cv.waitKey(100)
            if a == ord('q'):
                cv.destroyAllWindows()
                break
'''OBJ = pose_estimater()
img_test = cv.imread('./dataset/toolholder/images/toolholder9.jpg', 0)
imagelist = os.listdir('dataset/toolholder/images')
for i in imagelist:
    img = cv.imread('dataset/toolholder/images/'+i, 0)
    p = OBJ.estimate_pose(img)
    if p is not None:
        print(np.linalg.norm(p - np.array([[26.5], [24.0], [220]]), ord=2))'''
'''pt1 = np.array([37.8, 70.16, 1])
pt2 = np.array([257.13, 70.688, 1])
pt3 = np.array([248.8, 279.6, 1])
pt4 = np.array([39.6, 281.18, 1])

M = np.array([[ 8.78771481e-01, -1.74986412e-03,  2.17804793e+02],
 [ 1.06895080e-02,  8.55676506e-01,  2.20940648e+02],
 [-3.60026515e-06, -6.50585684e-06,  1.00000000e+00]])

[[1.08665566e+00 6.74684250e-02 3.83722713e+02]
 [1.84632309e-02 1.00873669e+00 1.53060739e+02]
 [2.54635746e-04 3.57427407e-05 1.00000000e+00]]
 
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
print('----------\n')'''
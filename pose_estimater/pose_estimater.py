import numpy as np
import cv2 as cv
import json
import os
import matplotlib.pyplot as plt
#import multiprocessing
import scipy.linalg as linalg
import math

class PoseEstimater():
    def __init__(self, _algorithm='SIFT', min_match=25):
        self.camera_matrix = None
        self.distor_matrix = None
        self.min_match = min_match
        self.algorithm = _algorithm
        self.kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        self.dataset = {}
        self.showmatchflag = 0
        self.transformpxel = None
        self.showmatchflag = 0
        self.img_query = {}
        self.match_img = None
        if _algorithm == 'SURF':
            self.detecter = cv.xfeatures2d.SURF_create(hessianThreshold=100, nOctaves=10, nOctaveLayers=2, extended=1, upright=0)
        elif _algorithm == 'SIFT':
            self.detecter = cv.xfeatures2d.SIFT_create(nfeatures=0, nOctaveLayers=3, contrastThreshold=0.05, edgeThreshold=10, sigma=0.8)
        #self.queue = multiprocessing.Queue()
        #self.show_match = multiprocessing.Process(target=self.show_match)


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
                _dataset['flag_point'] = [(wpt[0]+wpt[3])/2, (wpt[1]+wpt[4])/2, (wpt[2]+wpt[5])/2]
                self.dataset[dir] = _dataset
                self.img_query[dir] = cv.imread(_dataset_path+dir+'/images/'+dir+'.jpg')

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

    def pic_match(self, _img, _estimater_pose):
        img_test = _img
        d = 10000
        obj = ''
        #img_test = cv.filter2D(img_test, -1, self.kernel)
        #img_query = _img_query
        kp_test, des_test = self.detecter.detectAndCompute(img_test, None)
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv.FlannBasedMatcher(index_params, search_params)
        for _obj in self.dataset.keys():
            tmp_list = np.array(self.dataset[_obj]['flag_point'])
            tmp = np.linalg.norm(tmp_list-np.array(_estimater_pose[0:3]), 2)
            # print('obj:{}, distance:{}'.format(_obj, tmp))
            if tmp < d:
                d = tmp
                obj = _obj
        # print("choose {}".format(obj))
        des_query = self.dataset[obj]['des']
        kp_query = self.dataset[obj]['kp']
        try:
            matches = flann.knnMatch(des_query, des_test, k=2)
        except cv.error:
            return None, None
        good = []
        kp_good_match_query = []
        des_good_match_query = []
        for m, n in matches:
            if m.distance < 0.6 * n.distance:
                good.append(m)
                '''print('--------------------\n')
                print('m.imgIdx: {}\n'.format(m.imgIdx))
                print('m.queryIdx: {}\n'.format(m.queryIdx))
                print('m.trainIdx: {}\n'.format(m.trainIdx))
                print('kp_query: {}\n'.format(kp_query[m.queryIdx].pt))
                print('kp_test: {}\n'.format(kp_test[m.trainIdx].pt))'''
                kp_good_match_query.append(kp_query[m.queryIdx])
                des_good_match_query.append(des_query[m.queryIdx])
        # print('the num of finding featurs of query is {}\n'.format(len(des_query)))
        # print('the num of finding featurs of test is {}\n'.format(len(des_test)))
        # print('the num of finding matches is {}\n'.format(len(matches)))
        # print("good mathch of {}: {}".format(obj, len(good)))
        if len(good) > self.min_match:
            src_pts = np.float32([kp_good_match_query[i].pt for i in range(len(kp_good_match_query))]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_test[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
            matchesMask = mask.ravel().tolist()
            if M is not None and mask is not None:
                det = np.linalg.det(M)
                if det < 1.35 and det > 0.65:
                    pxel = self.dataset[obj]['wpixel'].reshape(-1, 1, 2)
                    pxel = cv.perspectiveTransform(pxel, M)
                    if self.showmatchflag == 1:
                        img_query = self.img_query[obj]
                        h, w = img_query.shape[0:2]
                        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
                        dst = cv.perspectiveTransform(pts, M)
                        #print('sss {}'.format(wpxel))
                        img_test = cv.polylines(img_test, [np.int32(dst)], True, 255, 1, cv.LINE_AA)
                        draw_params = dict(matchColor=(0, 255, 0),
                                           singlePointColor=None,
                                           matchesMask=matchesMask,
                                           flags=2)
                        mimg = cv.drawMatches(img_query, kp_query, img_test, kp_test, good, None, **draw_params)
                        tmppoint = pxel.reshape(-1, 1)
                        point = []
                        for i in range(0, len(tmppoint), 2):
                            point.append((int(tmppoint[i]+img_query.shape[1]), int(tmppoint[i + 1])))
                        img = mimg
                        for p in point:
                            img = cv.circle(img, p, 4, (255, 0, 0), -1)
                        # self.queue.put(img)
                        # plt.imshow(img)
                        # plt.show()
                        # cv.imwrite(str(self.index)+'.jpg', img)
                        self.match_img = img
                    return obj, pxel
        return None, None

    def estimate_pose(self, _img, estimater_pose):
        obj, _wpxel = self.pic_match(_img, estimater_pose)
        if obj is not None and _wpxel is not None:
            wpt = self.dataset[obj]['wpoint'].reshape(-1, 3)
            _wpxel = _wpxel.reshape(-1, 2)
            self.transformpxel = _wpxel.copy()
            # print(_wpxel, wpt)
            _wpxel[:, 0] = (_wpxel[:, 0]-self.camera_matrix[0, 2])*9.8 / 10
            _wpxel[:, 1] = (_wpxel[:, 1] - self.camera_matrix[1, 2]) * 9.8 / 10
            # print(_wpxel, wpt)
            # a = math.atan2(v[1], v[0]) / math.pi * 180
            # cross = np.cross(-_wpxel[0], v)
            # d = cross/np.linalg.norm(v, 2)
            # print(cross, d)
            _wpxel = np.c_[_wpxel, [[0],[0]]]
            # print(_wpxel)
            r = np.array([[-1,0,0],[0,1,0],[0,0,-1]])
            _wpxel_w = np.array([])
            _wpxel_w = np.dot(r, _wpxel[0])
            _wpxel_w = np.vstack((_wpxel_w, np.dot(r, _wpxel[1])))
            if wpt[0][1] > wpt[1][1]:
                head = _wpxel_w[0]
                tail = _wpxel_w[1]
                head_w = wpt[0]
                tail_w = wpt[1]
            else:
                head = _wpxel_w[1]
                tail = _wpxel_w[0]
                head_w = wpt[1]
                tail_w = wpt[0]
            v = head - tail
            # print('tial_w', tail_w)
            # print(v, _wpxel_w)
            a = math.atan2(v[1], v[0]) / math.pi * 180
            # print(a)
            cross = np.cross(-tail, v)
            d = cross/np.linalg.norm(v, 2)
            # print(d)
            # theta = math.atan2(v[1], v[0]) - math.atan2(-tail[1], -tail[0]) / math.pi * 180
            # print(theta)
            pose_y = tail_w[1] - d[2]
            v_y = np.dot(np.array([[0,-1,0],[1,0,0],[0,0,1]]), v)
            # print(v_y)
            cross = np.cross(-tail, v_y)
            d = cross / np.linalg.norm(v_y, 2)
            # print(d)
            pose_x = tail_w[0] + d[2]
            # print(self.camera_matrix)
          # print(self.transformpxel)
            pose_z = self.camera_matrix[0][0] * 1.1 * np.linalg.norm(head_w - tail_w, 2)/np.linalg.norm(self.transformpxel[0]-self.transformpxel[1]) + tail_w[2]
            # print(pose_z)
            pose = np.array([pose_x, pose_y,pose_z])
            # print('a:{}'.format(a))
            if self.showmatchflag == 1:
                cv.imwrite('/home/jakeluo/tello_project/log/match_img/'+str(pose_x)+'-'+str(pose_y)+'-'+str(pose_z)+'-'+str(a)+'.jpg', self.match_img)
            return pose, -a
            # # print('pose: {},{}'.format(pose_x, pose_y))
            # wpt = self.dataset[obj]['wpoint'].reshape(-1, 1, 3)
            # _wpxel = _wpxel.reshape(-1, 1, 2)
            # self.transformpxel = _wpxel
            # #wpxel = self.dataset[obj]['wpixel'].reshape(-1, 1, 2)
            # #print(self.camera_matrix)
            # #print(self.distor_matrix)
            # #print('transformed wpxl: \n{}'.format(_wpxel))
            # pnppara = dict(objectPoints=wpt,
            #                imagePoints=_wpxel,
            #                cameraMatrix=self.camera_matrix,
            #                distCoeffs=self.distor_matrix,
            #                useExtrinsicGuess=0,
            #                flags=cv.SOLVEPNP_ITERATIVE)
            # RR, rvec, tvec, inliers = cv.solvePnPRansac(**pnppara)
            # #print(rvec)
            # #print(RR)
            # print('inliers:{}'.format(inliers))
            # #print('tvec/n{}'.format(tvec/2))
            # if RR is True and len(inliers) >= 6:
            #     rotM = np.array(cv.Rodrigues(rvec)[0])
            #     R = rotM
            #     # print(R)
            #     sy = math.sqrt(R[0, 0] * R[0, 0] + R[0, 1] * R[0, 1])
            #     singular = sy < 1e-6
            #     if not singular:
            #         x = math.atan2(R[1, 2], R[2, 2])
            #         y = math.atan2(-R[0, 2], sy)
            #         z = math.atan2(math.sin(x)*R[2, 0]-math.cos(x)*R[1, 0], math.cos(x)*R[1, 1]-math.sin(x)*R[2, 1])
            #     else:
            #         x = math.atan2(-R[1, 2], R[1, 1])
            #         y = math.atan2(-R[2, 0], sy)
            #         z = 0
            #     # print('dst:', R)
            #     # print('x: {}\ny: {}\nz: {}'.format(x, y, z))
            #     # print('rvec:{}\n'.format(rvec))
            #     # print('tvec:{}\n'.format(tvec))
            #     #rotM = np.array(cv.Rodrigues(rvec)[0])
            #     # print('rotM {}\n)'.format(rotM))
            #     # print(-np.linalg.inv(rotM))
            #     pose = np.dot(np.linalg.inv(-rotM), tvec)
            #     #print(inliers)
            #     return pose, z *180/3.1416 +90
            # else:
            #     return None, None
        else:
            return None, None

    def show_match_start(self):
        self.showmatchflag = 1
        # self.show_match.start()

    def show_pic(self, _img):
        fig = plt.figure(figsize=(12, 10))
        plt.subplot(1, 1, 1).axis("off")
        plt.imshow(_img)
        plt.show()

    def showdataset(self, _obj='all'):
        obj = _obj
        if _obj=='all':
            for i in self.dataset.keys():
                print('-----------dataset-----------\n')
                print('----------{}----------\n'.format(i))
                for k in self.dataset[i].keys():
                    print(self.dataset[i][k])
        else:
            print('-----------dataset-----------\n')
            print('----------{}----------\n'.format(_obj))
            for k in self.dataset[_obj].keys():
                print(self.dataset[_obj][k])

    def show_match(self):
        while True:
            if not self.queue.empty():
                f = self.queue.get()
                cv.imshow('match_img', f)
            a = cv.waitKey(100)
            if a == ord('q'):
                cv.destroyAllWindows()
                break

    def show_transedpxel(self, _img_test):
        if self.transformpxel is not None:
            tmppoint = self.transformpxel.reshape(-1)
            point = []
            for i in range(0, len(tmppoint), 2):
                point.append((int(tmppoint[i]), int(tmppoint[i+1])))
            img = _img_test
            for p in point:
                img = cv.circle(img, p, 2, (255, 0, 0), -1)
            plt.imshow(img)
            plt.show()

    def modifydata(self, obj, pixel=True, point=True):
        self.showdataset()
        wpixel = np.array([])
        wpoint = np.array([])
        for i in range(int(self.dataset[obj]['wpoint'].size/3)):
            if pixel==True:
                wpxlx = input('input the x of No.{} wpixel:'.format(i + 1))
                wpxly = input('input the y of No.{} wpixel:'.format(i + 1))
                wpxl = np.array([float(wpxlx), float(wpxly)])
                wpixel = np.append(wpixel, wpxl)
            if point==True:
                wptx = input('input the x of No.{} wpoint:'.format(i + 1))
                wpty = input('input the y of No.{} wpoint:'.format(i + 1))
                wptz = input('input the z of No.{} wpoint:'.format(i + 1))
                wpt = np.array([float(wptx), float(wpty), float(wptz)])
                wpoint = np.append(wpoint, wpt)
        if pixel==True:
            self.save_2_npy('/home/jakeluo/tello_project/pose_estimater/dataset/' + obj + '/wpixel.npy', wpixel)
        if point==True:
            self.save_2_npy('/home/jakeluo/tello_project/pose_estimater/dataset/' + obj + '/wpoint.npy', wpoint)


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
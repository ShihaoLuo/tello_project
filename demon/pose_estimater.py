import tensorflow as tf
import numpy as np
from PIL import Image
#from matplotlib import pyplot as plt
import os
import sys
import cv2 as cv
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
from depthmotionnet.networks_original import *

class Demon():
    def __init__(self):
        #self.examples_dir = os.path.dirname(__file__)
        #self.weights_dir = os.path.join(self.examples_dir, '..', 'weights')
        #self.sys.path.insert(0, os.path.join(examples_dir, '..', 'python'))
        if tf.test.is_gpu_available(True):
            self.data_format = 'channels_first'
        else:  # running on cpu requires channels_last data format
            self.data_format = 'channels_last'
        self.gpu_options = tf.GPUOptions()
        self.gpu_options.per_process_gpu_memory_fraction = 0.8
        self.session = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=self.gpu_options))
        self.bootstrap_net = BootstrapNet(self.session, self.data_format)
        self.iterative_net = IterativeNet(self.session, self.data_format)
        self.refine_net = RefinementNet(self.session, self.data_format)
        self.session.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        self.saver.restore(self.session, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'weights/demon_original'))
        self.camera_M = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)),'camera_matrix_tello.npy'))
        self.distor_M = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)),'distor_matrix_tello.npy'))

    def process_pic(self, img):
        img1 = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        size = (img1.shape[1], img1.shape[0])

        K = np.array([[0.89115971, 0, 0.5],
                      [0, 1.18821287, 0.5],
                      [0, 0, 1 / 1000]]) * 1000
        camera_M_inv = np.linalg.inv(self.camera_M)
        camera_M_inv[2, 2] = 1
        M = np.dot(K, camera_M_inv)

        scaled_camera_matrix, roi = cv.getOptimalNewCameraMatrix(
            self.camera_M, self.distor_M, size, 1, size
        )
        img1 = cv.undistort(img1, self.camera_M, self.distor_M, None, scaled_camera_matrix)
        img1 = cv.warpPerspective(img1, M, size)
        img1 = Image.fromarray(img1)
        return img1

    def prepare_input_data(self, img1, img2, data_format):
        """Creates the arrays used as input from the two images."""
        # scale images if necessary
        if img1.size[0] != 256 or img1.size[1] != 192:
            img1 = img1.resize((256, 192))
        if img2.size[0] != 256 or img2.size[1] != 192:
            img2 = img2.resize((256, 192))
        img2_2 = img2.resize((64, 48))

        # transform range from [0,255] to [-0.5,0.5]
        img1_arr = np.array(img1).astype(np.float32) / 255 - 0.5
        img2_arr = np.array(img2).astype(np.float32) / 255 - 0.5
        img2_2_arr = np.array(img2_2).astype(np.float32) / 255 - 0.5

        if data_format == 'channels_first':
            img1_arr = img1_arr.transpose([2, 0, 1])
            img2_arr = img2_arr.transpose([2, 0, 1])
            img2_2_arr = img2_2_arr.transpose([2, 0, 1])
            image_pair = np.concatenate((img1_arr, img2_arr), axis=0)
        else:
            image_pair = np.concatenate((img1_arr, img2_arr), axis=-1)

        result = {
            'image_pair': image_pair[np.newaxis, :],
            'image1': img1_arr[np.newaxis, :],  # first image
            'image2_2': img2_2_arr[np.newaxis, :],  # second image with (w=64,h=48)
        }
        return result

    def estimatepose(self, img1, img2):
        img1 = self.process_pic(img1)
        img2 = self.process_pic(img2)
        input_data = self.prepare_input_data(img1, img2, self.data_format)
        result = self.bootstrap_net.eval(input_data['image_pair'], input_data['image2_2'])
        for i in range(3):
            result = self.iterative_net.eval(
                input_data['image_pair'],
                input_data['image2_2'],
                result['predict_depth2'],
                result['predict_normal2'],
                result['predict_rotation'],
                result['predict_translation']
            )
        rotation = result['predict_rotation']
        translation = result['predict_translation']
        result = self.refine_net.eval(input_data['image1'], result['predict_depth2'])
        rot = cv.Rodrigues(rotation)[0]
        return rot, translation
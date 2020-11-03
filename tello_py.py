#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 22:11:47 2020

@author: jake
"""

import tello_controller
import tello_video
import multiprocessing
import numpy as np
#import logging
from pose_estimater import  pose_estimater
import cv2 as cv
from control_panel import control_panel
import time

def process_frame(_video, _pose_estimater):
    global pose
    #img_query = cv.imread('pose_estimater/dataset/post/images/post.jpg', 0)
    img_query = cv.imread('pose_estimater/dataset/post/images/post.jpg', 0)
    #log = './log_pose/pose.log'
    #logging.basicConfig(filename=log, level=logging.DEBUG, format='%(asctime)s %(message)s', datefmt='%d/%m/%Y %H:%M:%S')
    while True:
        _frame = _video.get_frame()
        if _frame is not None:
            pose, yaw = _pose_estimater.estimate_pose(img_query, _frame)
            if pose is not None:
                print("Pose in the world is {} {}".format(pose, yaw))
                #logging.info("\n{}".format(pose))


controller = tello_controller.Tell_Controller()
pose_estimater = pose_estimater.PoseEstimater('SIFT', 15)
#pose_estimater.show_match_start()
pose_estimater.loaddata('pose_estimater/dataset/')
frame = None
pose = np.array([])
move_command = ['forward 100', 'left 100', 'right 100', 'back 100', 'left 100', 'right 100']

try:
    controller.scan(1)
    video = tello_video.Tello_Video(controller.tello_list)
    pose_thread = multiprocessing.Process(target=process_frame, args=(video, pose_estimater,))
    pose_thread.start()
    controller.command("battery_check 20")
    controller.command("correct_ip")
    for i in range(len(controller.sn_list)):
        controller.command(str(i + 1) + "=" + controller.sn_list[i])
    ctr = control_panel.CtrPanel(controller)
    controller.command('*>setfps high')
    controller.command('*>setresolution high')
    controller.command('setbitrate 5')
    controller.command("*>takeoff")
    controller.command('*>up 170')
    controller.command("*>streamon")
    #controller.command("wait 10")
    ctr.start()
    #controller.control_mode()
    #for i in range(12):
     #   controller.command('*>'+move_command[i%6])
      #  controller.command('wait 5')
    #controller.command("wait 20")
    '''controller.command("*>takeoff")
    controller.command("*>up 20")
    controller.command("wait 5")
    controller.command("*>back 100")
    controller.command("wait 5")
    controller.command("*>left 100")
    controller.command("wait 5")
    controller.command("*>forward 100")
    controller.command("wait 5")
    controller.command("*>right 100")'''
    # start_time = time.time()
    # end_time = time.time()
    # controller.command("wait 20")
    controller.command("*>streamoff")
    controller.command("*>land")
    # time.sleep(50)

    controller.save_log(controller.manager)
    controller.manager.close()
    video.close()
    pose_thread.terminate()
except KeyboardInterrupt:
    print ('[Quit_ALL]Multi_Tello_Task got exception. \
           Sending land to all drones...\n')
    for ip in controller.manager.tello_ip_list:
        controller.manager.socket.sendto('streamoff'.encode('utf-8'),
                                         (ip, 8889))
        controller.manager.socket.sendto('land'.encode('utf-8'), (ip, 8889))
    controller.save_log(controller.manager)
    controller.manager.close()
    video.close()

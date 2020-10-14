#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 22:11:47 2020

@author: jake
"""

import tello_controller
import tello_video
import multiprocessing
import time
import numpy as np
import logging
from pose_estimater import  pose_estimater


def process_frame(_video, _pose_estimater):
    global pose
    log = './log_pose/pose.log'
    logging.basicConfig(filename=log,
                        level=logging.DEBUG,
                        format='%(asctime)s %(message)s',
                        datefmt='%d/%m/%Y %H:%M:%S')
    while True:
        _frame = _video.get_frame()
        if _frame is not None:
            pose = _pose_estimater.estimate_pose(_frame)
            if pose is not None:
                print("Pose in the world is {}\n".format(pose))
                logging.info("\nPose in the world is {}\n".format(pose))
        #time.sleep(0.025)

controller = tello_controller.Tell_Controller()
# marker_detecter = marker_detecter.Marker_Manager()
pose_estimater = pose_estimater.PoseEstimater()
pose_estimater.loaddata('pose_estimater/dataset/')
frame = None
pose = np.array([])
try:
    controller.scan(1)
    video = tello_video.Tello_Video(controller.tello_list)
    pose_thread = multiprocessing.Process(target=process_frame, args=(video,pose_estimater,))
    pose_thread.start()
    controller.command("battery_check 20")
    controller.command("correct_ip")
    for i in range(len(controller.sn_list)):
        controller.command(str(i + 1) + "=" + controller.sn_list[i])
    controller.command('*>setfps middle')
    controller.command('*>setresolution low')
    controller.command('setbitrate 5')
    controller.command("*>streamon")
    controller.command("wait 50")
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
    controller.command("*>land")
    # time.sleep(50)
    controller.command("*>streamoff")
    controller.save_log(controller.manager)
    controller.manager.close()
    video.close()
    pose_thread.terminate()
except KeyboardInterrupt:
    print ('[Quit_ALL]Multi_Tello_Task got exception. \
           Sending land to all drones...\n')
    for ip in controller.manager.tello_ip_list:
        controller.manager.socket.sendto('streamoff'.encode('utf-8'),
                                         (ip, 9001))
        controller.manager.socket.sendto('land'.encode('utf-8'), (ip, 9001))
    controller.save_log(controller.manager)
    controller.manager.close()
    video.close()

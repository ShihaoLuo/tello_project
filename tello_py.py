#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 22:11:47 2020

@author: jake
"""

import tello_controller
import tello_video
# import cv2 as cv
# import marker_detecter
import threading
import time
import numpy as np
import logging


'''def process_frame(_video, _marker_detecter):
    global pose
    log = './log_pose/pose.log'
    logging.basicConfig(filename=log,
                        level=logging.DEBUG,
                        format='%(asctime)s %(message)s',
                        datefmt='%d/%m/%Y %H:%M:%S')

    point_world = np.array([[0.223, 0.0, 0.0],
                            [0.0, 0.0, 0.0],
                            [0.0, 0.223, 0.0],
                            [0.223, 0.223, 0.0]])
    while True:
        frame = _video.get_frame()
        if frame is None:
            pass
        else:
            # print("get a frame in the marker thread.")
            # _marker_detecter.drawdetectedmarker(frame)
            _marker_detecter.marker_detect(frame)
            if _marker_detecter.markerIds is not None:
                pose = _marker_detecter.estimate_pose(point_world)
                print("Pose in the world is {}\n".format(pose))
                logging.info("\nPose in the world is {}\n".format(pose))
        time.sleep(0.075)'''

'''def go_rectangle(pose):
    path = np.array([[0, 0, 1.5],
                     [1, 0, 1.5],
                     [1, 0, 2.5],
                     [0, 0, 2.5]])
    controller.command()'''


controller = tello_controller.Tell_Controller()
# marker_detecter = marker_detecter.Marker_Manager()
frame = None
pose = np.array([])

try:
    controller.scan(1)
    video = tello_video.Tello_Video(controller.tello_list)
    controller.command("battery_check 20")
    controller.command("correct_ip")
    for i in range(len(controller.sn_list)):
        controller.command(str(i + 1) + "=" + controller.sn_list[i])
    controller.command("*>streamon")
    controller.command("*>takeoff")
    controller.command("*>up 100")
    controller.command("wait 5")
    controller.command("*>back 100")
    controller.command("wait 5")
    controller.command("*>left 100")
    controller.command("wait 5")
    controller.command("*>forward 100")
    controller.command("wait 5")
    controller.command("*>right 100")
    # start_time = time.time()
    # end_time = time.time()
    # controller.command("wait 20")
    controller.command("*>land")
    # time.sleep(50)
    controller.command("*>streamoff")
    controller.save_log(controller.manager)
    controller.manager.close()
    video.close()
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

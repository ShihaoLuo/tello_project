#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 22:11:47 2020

@author: jake
"""

import tello_controller
import tello_video
import numpy as np
import cv2 as cv
import time
import Scheduler


def process_frame(_video, _pose_estimater):
    global pose
    img_query = cv.imread('pose_estimater/dataset/post/images/post.jpg', 0)
    while True:
        _frame = _video.get_frame()
        if _frame is not None:
            pose, yaw = _pose_estimater.estimate_pose(_frame)
            if pose is not None:
                print("Pose in the world is {} {}".format(pose, yaw))
                pos[0] = pose[0]
                pos[1] = pose[2]
                pos[2] = yaw


def updatepos(_last_cmd, _last_pose, _Video, _pose_estimater):
    img = _Video.get_frame()
    pose = np.array([0, 0, 0, 0])
    if img is not None:
        _pose, yaw = _pose_estimater.estimate_pose(img)
        if _pose is not None:
            print('in img1')
            time.sleep(2.5)
            while True:
                img = _Video.get_frame()
                if img is not None:
                    break
            _pose, yaw = _pose_estimater.estimate_pose(img)
            if _pose is not None:
                print('in img2')
                pose[0] = _pose[0]
                pose[1] = _pose[1]
                pose[2] = _pose[2]
                pose[3] = yaw
                return pose
    if 'ccw' in _last_cmd:
        angle = float(_last_cmd.partition(' ')[2])
        print('angle:{}'.format(angle))
        pose[3] = angle + _last_pose[3]
        pose[0:3] = _last_pose[0:3]
    elif 'go' in _last_cmd:
        tmp = _last_cmd.split(' ')[1:4]
        tmp = [int(i) for i in tmp]
        alpha = _last_pose[3] * 3.1416 / 180
        M = np.array([[np.cos(alpha), np.sin(alpha), 0],
                      [-np.sin(alpha), np.cos(alpha), 0],
                      [0, 0, 1]])
        tmp = np.dot(np.linalg.inv(M), tmp)
        tmp = np.append(tmp, 0)
        pose = _last_pose + tmp
    else:
        pose = _last_pose
    return pose


controller = tello_controller.Tell_Controller()
#pose_estimater.modifydata('post', False, False)
frame = None
path1 = [[-350, 0, 100, 0],
         [-250, 0, 100, 0],
        [-250, 0, 100, 90],
        [-250, 100, 100, 90],
        [-250, 200, 100, 90],
        [-250, 300, 100, 90],
        [-250, 400, 100, 90],
        [-250, 500, 100, 90],
        [-250, 600, 100, 90],
        [-250, 600, 100, 180],
        [-350, 600, 100, 180],
        [-450, 600, 100, 180],
        [-550, 600, 100, 180],
        [-550, 600, 100, 270],
        [-550, 500, 100, 270],
        [-550, 400, 100, 270],
        [-550, 300, 100, 270],
        [-550, 200, 100, 270],
        [-550, 100, 100, 270],
        [-550, 0, 100, 270],
        [-550, 0, 100, 0],
        [-450, 0, 100, 0]]
path2 = [[-350, 0, 100, 0],
         [-250, 0, 100, 0],
        [-250, 0, 100, 90],
        [-250, 100, 100, 90],
        [-250, 200, 100, 90],
        [-250, 300, 100, 90],
        [-250, 400, 100, 90],
        [-250, 500, 100, 90],
        [-250, 600, 100, 90],
        [-250, 600, 100, 180],
        [-350, 600, 100, 180],
        [-450, 600, 100, 180],
        [-550, 600, 100, 180],
        [-550, 600, 100, 270],
        [-550, 500, 100, 270],
        [-550, 400, 100, 270],
        [-550, 300, 100, 270],
        [-550, 200, 100, 270],
        [-550, 100, 100, 270],
        [-550, 0, 100, 270],
        [-550, 0, 100, 0],
        [-450, 0, 100, 0]]
path3 = [[-350, 0, 100, 0],
         [-250, 0, 100, 0],
        [-250, 0, 100, 90],
        [-250, 100, 100, 90],
        [-250, 200, 100, 90],
        [-250, 300, 100, 90],
        [-250, 400, 100, 90],
        [-250, 500, 100, 90],
        [-250, 600, 100, 90],
        [-250, 600, 100, 180],
        [-350, 600, 100, 180],
        [-450, 600, 100, 180],
        [-550, 600, 100, 180],
        [-550, 600, 100, 270],
        [-550, 500, 100, 270],
        [-550, 400, 100, 270],
        [-550, 300, 100, 270],
        [-550, 200, 100, 270],
        [-550, 100, 100, 270],
        [-550, 0, 100, 270],
        [-550, 0, 100, 0],
        [-450, 0, 100, 0]]
num = 1

try:
    controller.scan(num)
    video = tello_video.Tello_Video(controller.tello_list)
    controller.command("battery_check 20")
    controller.command("correct_ip")
    for i in range(len(controller.sn_list)):
        controller.command(str(i + 1) + "=" + controller.sn_list[i])
    Scheduler = Scheduler.Scheduler(controller, video)
    for i in range(num):
        Scheduler.init_path(i+1, eval('path'+str(i+1)), [-450, 0, 0, 0])
    #Scheduler.init_path(1, 2, path2, [-450, 0, 0, 0])
    #Scheduler.drone_init()
    #controller.command("1>takeoff")
    #controller.command('1>up 170')
    Scheduler.start()
    print('after thread start_______________________')
    now = time.time()
    now_run = {}
    for i in range(num):
        now_run[i] = now + i * 5
    while True:
        # print('schedule thread living...', Scheduler.get_schedule_handle().isAlive())
        # for i in range(len(controller.sn_list)):
        #     print('run thread living...', i+1, Scheduler.get_runthread_handle()[i+1].isAlive())
        for i in range(num):
            if time.time() - now_run[i] > 15:
                Scheduler.update_path(i + 1, eval('path' + str(i + 1)))
                now_run[i] = time.time()
        if time.time() - now > 200:
            print('main thread break')
            break
        time.sleep(0.5)
    Scheduler.stop_thread()
    # for i in range(1):
    #     controller.command('1>battery?')
    #     for target in path:
    #         print("--------------------------")
    #         print("target:{}".format(target))
    #         theta = target[3] - pose[3]
    #         if abs(theta) > 30:
    #             cmd = 'ccw ' + str(theta)
    #             controller.command("1>"+cmd)
    #             pose = updatepos(cmd, pose, video, pose_estimater)
    #             print("Pose in the drone_world is {}".format(pose))
    #         if np.linalg.norm(target[0:3] - pose[0:3]) < 50:
    #             pass
    #         else:
    #             alpha = pose[3] * 3.1416 / 180
    #             #print("alpha:{}".format(alpha))
    #             M = np.array([[np.cos(alpha), np.sin(alpha), 0],
    #                           [-np.sin(alpha), np.cos(alpha), 0],
    #                           [0, 0, 1]])
    #             #print("M: {}".format(M))
    #             tmp = target[0:3]-pose[0:3]
    #             tmp = np.dot(M, tmp)
    #             #tmp = [int(i) for i in tmp]
    #             tmp = np.append(tmp, 100)
    #             tmp = [int(i) for i in tmp]
    #             tmp = [str(i) for i in tmp]
    #             cmd = 'go ' + ' '.join(tmp)
    #             controller.command('1>'+cmd)
    #             pose = updatepos(cmd, pose, video, pose_estimater)
    #             print("Pose in the drone_world is {}".format(pose))
    print('main dying...')
    controller.command("*>land")
    controller.command("*>streamoff")
    controller.save_log(controller.manager)
    controller.manager.close()
    video.close()
    #pose_thread.terminate()
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

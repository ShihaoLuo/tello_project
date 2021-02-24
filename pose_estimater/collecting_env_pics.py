#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 09:33:51 2020

@author: jake
"""

import tello_video
import tello_controller
import time
import os

controller = tello_controller.Tell_Controller()
controller.scan(1)
controller.command("battery_check 20")
controller.command("correct_ip")
video = tello_video.Tello_Video(controller.tello_list)

name = 'post2'
num = 10

pic_folder = './dataset/'+name
if not os.path.exists(pic_folder):
    os.mkdir(pic_folder)
pic_folder = './dataset/'+name+'/images/'
if not os.path.exists(pic_folder):
    os.mkdir(pic_folder)


init_command = ['setfps high', 'setresolution high', 'setbitrate 5', 'streamon',  'takeoff', 'up 130']
move_command = ['right 50', 'left 50', 'left 50', 'right 50', 'back 50']

try:
    for i in range(len(controller.sn_list)):
        controller.command(str(i + 1) + "=" + controller.sn_list[i])
    for init_c in init_command:
        controller.command('*>'+init_c)
    for i in range(num):
        controller.command('*>'+move_command[i%5])
        video.take_pic(pic_folder + name + str(i) + '.jpg')
        controller.command('*wait 1')
    controller.command('*>land')
    controller.save_log(controller.manager)
    controller.manager.close()
    video.close()
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

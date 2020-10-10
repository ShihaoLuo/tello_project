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

name = 'toolholder'
num = 25

pic_folder = './dataset/toolholder/images/'
if not os.path.exists('dataset/toolholder/images'):
    os.mkdir('dataset/toolholder/images')


init_command = ['streamon']
move_command = ['right 100']*8

try:
    for i in range(len(controller.sn_list)):
        controller.command(str(i + 1) + "=" + controller.sn_list[i])
    for init_c in init_command:
        controller.command('*>'+init_c)

    time.sleep(10)
    for i in range(num):
        video.take_pic(pic_folder+name+str(i+19)+'.jpg')
        time.sleep(2)
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

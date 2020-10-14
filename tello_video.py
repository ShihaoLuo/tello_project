#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 18:16:23 2020

@author: jakeluo
"""
import socket
import multiprocessing
import numpy as np
import libh264decoder
import cv2
import time


class Tello_Video:
    def __init__(self, tello_list):
        # the tello list can be provided by tello_cotroller.tell_list after using controller.scan()
        print("tello_video_ instance init....")
        self.tello_list = tello_list
        self.video_port_base = 21010
        self.local_ip = ''
        self.local_video_port = {}
        self.sock_video = {}
        self.queue = {}
        self.receive_video_thread = {}
        self.tello_ip = ''
        for tello in tello_list:
            self.local_video_port[tello.tello_ip] = self.video_port_base + int(
                str(tello.tello_ip).split('.')[3])
        for tello in tello_list:
            self.sock_video[tello.tello_ip] = socket.socket(
                socket.AF_INET, socket.SOCK_DGRAM)
            self.sock_video[tello.tello_ip].setsockopt(
                socket.SOL_SOCKET, socket.SO_RCVBUF, 512*1024)
            self.sock_video[tello.tello_ip].setsockopt(
                socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.sock_video[tello.tello_ip].bind(
                (self.local_ip, 11111)) # self.local_video_port[tello.tello_ip]))
        self.tello_ip_list = []
        self.video_cur_ip = ''
        self.res_string = ""
        self.lock = multiprocessing.Lock()
        for tello in tello_list:
            self.queue[tello.tello_ip] = multiprocessing.Queue(10)
        self.h264decoder = libh264decoder.H264Decoder()
        # self.res_frame_list = []

        for tello in tello_list:
            self.receive_video_thread[tello.tello_ip] = multiprocessing.Process(
                target=self._receive_video_thread, args=(tello.tello_ip,))
            self.receive_video_thread[tello.tello_ip].start()
        '''self.receive_video_thread = threading.Thread(target=self._receive_video_thread)
        self.receive_video_thread.setDaemon(1)
        # self.receive_video_thread.start()
        self.receive_video_thread2 = threading.Thread(target=self._receive_video_thread2)
        self.receive_video_thread2.setDaemon(1)'''

        self.show_video_thread = multiprocessing.Process(target=self.show_pic)
        # self.show_video_thread.setDaemon(1)
        self.show_video_thread.start()

        # self.decode_thread = threading.Thread(target=self._h264_decode)
        # self.decode_thread.setDaemon(1)
        # self.decode_thread.start()

    def close(self):

        for tello in self.tello_list:
            self.sock_video[tello.tello_ip].close()
            self.receive_video_thread[tello.tello_ip].terminate()
            self.show_video_thread.terminate()
        cv2.destroyAllWindows()
        # self.stop_thread()

    def _receive_video_thread(self, tello_ip):
        pack_data = ""
        print("receive video thread start....")
        while True:
            try:
                # print("in the receive video thread while loop...")
                res_string, ip = self.sock_video[tello_ip].recvfrom(2000)
                pack_data += res_string
                if len(res_string) != 1460:
                    # print("The size of packet data is %d.\n" % len(pack_data))
                    self._h264_decode(pack_data, tello_ip)
                    # self.Queue_res_buf.put(self.res_string)
                    pack_data = ""
            except socket.error as exc:
                print("Caught exception socket.error(video_thread): %s" % exc)

    def _h264_decode(self, packet_data, tello_ip):
        """
        decode raw h264 format data from Tello

        :param packet_data: raw h264 data array

        :return: a list of decoded frame
        """
        frames = self.h264decoder.decode(packet_data)
        for frame_data in frames:
            (frame, w, h, ls) = frame_data
            if frame is not None:
                # print ('frame size %i bytes, w %i, h %i, line_size %i' % (len(frame), w, h, ls))
                frame = np.fromstring(frame, dtype=np.ubyte, count=len(frame), sep='')
                frame = (frame.reshape((h, ls / 3, 3)))
                frame = frame[:, :w, :]
                self.queue[tello_ip].put(frame)

    def set_tello_ip(self, tello_ip):
        self.tello_ip = tello_ip

    def thread_start(self):
        for tello in self.tello_list:
            self.receive_video_thread[tello.tello_ip].start()

    '''def _async_raise(self, tid, exctype):
        tid = ctypes.c_long(tid)
        if not inspect.isclass(exctype):
            exctype = type(exctype)
        res = ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, ctypes.py_object(exctype))
        if res == 0:
            raise ValueError("invalid thread id")
        elif res != 1:
            # """if it returns a number greater than one, you're in trouble,
            # and you should call it again with exc=NULL to revert the effect"""
            ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, None)
            raise SystemError("PyThreadState_SetAsyncExc failed")

    def stop_thread(self):
        self._async_raise(self.receive_video_thread1.ident, SystemExit)
        self._async_raise(self.receive_video_thread2.ident, SystemExit)'''

    def show_pic(self):
        # frame_dict = {}.fromkeys(self.tello_ip_list,bytes())
        print("show pic thread start.\n")
        while True:
            a = cv2.waitKey(2)
            if a == 113:
                cv2.destroyAllWindows()
                break
            '''for tello in self.tello_list:
                flag.append(self.queue[tello.tello_ip].empty())
            tmp = set(flag)
            if (flag[0] == 0) and (len(tmp) == 1):
                for tello in self.tello_list:
                    f[tello.tello_ip] = self.queue[tello.tello_ip].get()
                    cv2.imshow(tello.tello_ip, f[tello.tello_ip])
            flag = []'''
            for tello in self.tello_list:
                if not self.queue[tello.tello_ip].empty():
                    f = self.queue[tello.tello_ip].get()
                    cv2.imshow(tello.tello_ip, f)
                #time.sleep(0.1)

    def take_pic(self, pic_name):
        mat = None
        # print("the saving addr is %s"%pic_name)
        if not self.queue[self.tello_list[0].tello_ip].empty():
            mat = self.queue[self.tello_list[0].tello_ip].get()
        # print(f)
        cv2.imwrite(pic_name, mat)

    def get_frame(self):
        if not self.queue[self.tello_list[0].tello_ip].empty():
            return self.queue[self.tello_list[0].tello_ip].get()
        else:
            return None

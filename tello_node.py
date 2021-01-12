# -*- coding: utf-8 -*-
# @Time    : 2021/1/1 下午5:02
# @Author  : JakeShihao Luo
# @Email   : jakeshihaoluo@gmail.com
# @File    : tello_node.py
# @Software: PyCharm

import socket
import time
import multiprocessing
import h264decoder
from pose_estimater.pose_estimater import *


class TelloNode:
    def __init__(self, tello_info, res_flag, main_flag):
        self.tello_ip = tello_info[0]
        self.ctr_port = tello_info[1]
        self.video_port = tello_info[2]
        self.ctr_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.ctr_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.ctr_socket.bind(('', self.ctr_port))
        self.pose = multiprocessing.Queue()
        self.video_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.video_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 512 * 1024)
        self.video_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.video_socket.bind(('', self.video_port))
        self.h264decoder = h264decoder.H264Decoder()
        self.queue = multiprocessing.Queue()
        self.pose_estimater = PoseEstimater('SIFT', 15)
        self.pose_estimater.loaddata('pose_estimater/dataset/')
        self.path = multiprocessing.Queue()
        self.video_flag = 1
        self.run_thread_flag = multiprocessing.Value('i', 0)
        self.update_path_flag = multiprocessing.Value('i', 0)
        self.Res_flag = res_flag[self.tello_ip]
        self.cmd = multiprocessing.Array('c', 20)
        self.cmd_event = multiprocessing.Event()
        self.update_path_event = multiprocessing.Event()
        self.main_flag = main_flag

    def get_thread_flag(self):
        return self.run_thread_flag

    def init_path(self, path, pose):
        tmp = np.array(path)
        for t in tmp:
            self.path.put(t)
        self.pose.put(pose)

    def update_path(self, path):
        update_path_thread = multiprocessing.Process(target=self._update_path, args=(path,))
        update_path_thread.start()

    def _update_path(self, path):
        self.update_path_event.clear()
        print('updating the path______________________________________', self.tello_ip)
        tmp = np.array(path)
        d = np.array([])
        pose = self.pose.get()
        self.pose.put(pose)
        for t in tmp:
            d = np.append(d, np.linalg.norm(np.array(pose[0:3]) - t[0:3], 2))
        a = np.argmin(d)
        while self.path.empty() is False:
            self.path.get()
        if a == len(tmp)-1:
            for t in tmp:
                self.path.put(t)
        else:
            for t in tmp[a + 1:]:
                self.path.put(t)
        print('finished updating the path_________________________________', self.tello_ip)
        self.update_path_event.set()

    def _h264_decode(self, packet_data):
        frames = self.h264decoder.decode(packet_data)
        for frame_data in frames:
            (frame, w, h, ls) = frame_data
            if frame is not None:
                frame = np.fromstring(frame, dtype=np.ubyte, count=len(frame), sep='')
                frame = (frame.reshape((h, ls // 3, 3)))
                frame = frame[:, :w, :]
                while self.queue.qsize() >= 2:
                    self.queue.get()
                self.queue.put(frame)

    def _receive_video_thread(self):
        pack_data = ''
        print("receive video thread start....")
        while True:
            try:
                res_string, ip = self.video_socket.recvfrom(2000)
                pack_data += res_string.hex()
                if len(res_string) != 1460:
                    self._h264_decode(bytes.fromhex(pack_data))
                    pack_data = ''
                if self.run_thread_flag.value == 1 or self.main_flag.value == 1:
                    time.sleep(1)
                    # self.video_socket.close()
                    print('video thread die..')
                    break
            except socket.error as exc:
                print("Caught exception socket.error(video_thread): %s in node" % exc)
                break
        print('video thread died...')

    # def update_res(self):
    #     self.cmd_res.put(1)

    def send_command(self, command):
        if command != '' and command != '\n':
            _command = command.rstrip()
            if '//' in _command:
                pass
            elif '>' in _command:
                action = str(_command.partition('>')[2])
                self.ctr_socket.sendto(action.encode('utf-8'), (self.tello_ip, 8889))
            elif 'wait' in _command:
                wait_time = float(_command.partition('wait')[2])
                action = 'command'
                while True:
                    cnt = wait_time - 5
                    if cnt > 0:
                        self.ctr_socket.sendto(action.encode('utf-8'), (self.tello_ip, 8889))
                        time.sleep(5)
                    else:
                        self.ctr_socket.sendto(action.encode('utf-8'), (self.tello_ip, 8889))
                        time.sleep(wait_time)
                        break
                    wait_time = cnt

    def run_thread(self):
        print('run thread start....')
        video_thread = multiprocessing.Process(target=self._receive_video_thread, daemon=True)
        video_thread.start()
        cmd_thread = multiprocessing.Process(target=self.update_cmd, daemon=True)
        cmd_thread.start()
        time.sleep(1)
        with self.cmd.get_lock():
            self.cmd.value = b'>command'
        self.cmd_event.set()
        while True:
            # print("video thread is alive,", video_thread.is_alive())
            # print("cmd thread is alive,", cmd_thread.is_alive())
            self.cmd_event.wait(5)
            if self.run_thread_flag.value == 1 or self.main_flag.value == 1:
                time.sleep(1)
                break
            with self.cmd.get_lock():
                cmd = self.cmd.value.decode()
            self.send_command(cmd)
            print('in run thread, send ', cmd)
            self.cmd_event.clear()
            # if cmd_thread.is_alive() is False:
            #     print('update_cmd thread is died...')
        print('run thread died...')

    def run(self):
        run_thread = multiprocessing.Process(target=self.run_thread)
        run_thread.start()

    def update_pos(self, _last_cmd, _last_pose):
        img = None
        pose = np.array([0, 0, 0, 0])
        if self.queue.empty() is False:
            img = self.queue.get()
        if img is not None:
            _pose, yaw = self.pose_estimater.estimate_pose(img)
            if _pose is not None:
                print('in img1')
                time.sleep(2.5)
                while self.queue.empty() is True:
                    time.sleep(0.01)
                img = self.queue.get()
                _pose, yaw = self.pose_estimater.estimate_pose(img)
                if _pose is not None:
                    print('in img2')
                    pose[0] = _pose[0]
                    pose[1] = _pose[1]
                    if _pose[2] == 0:
                        pose[2] = _last_pose[2]
                    else:
                        pose[2] = _pose[2]
                    pose[3] = yaw
                    print('update pose of:', self.tello_ip, pose)
                    return pose
        if '>ccw' in _last_cmd:
            angle = float(_last_cmd.partition(' ')[2])
            # print('angle:{}'.format(angle))
            pose[3] = angle + _last_pose[3]
            if pose[3] >= 360:
                pose[3] -= 360
            pose[0:3] = _last_pose[0:3]
        elif '>cw' in _last_cmd:
            angle = float(_last_cmd.partition(' ')[2])
            print('angle:{}'.format(angle))
            pose[3] = _last_pose[3] - angle
            if pose[3] >= 360:
                pose[3] -= 360
            pose[0:3] = _last_pose[0:3]
        elif '>go' in _last_cmd:
            tmp = _last_cmd.split(' ')[1:4]
            tmp = [int(i) for i in tmp]
            alpha = _last_pose[3] * 3.1416 / 180
            m = np.array([[np.cos(alpha), np.sin(alpha), 0],
                          [-np.sin(alpha), np.cos(alpha), 0],
                          [0, 0, 1]])
            tmp = np.dot(np.linalg.inv(m), tmp)
            tmp = np.append(tmp, 0)
            pose = _last_pose + tmp
        elif '>takeoff' in _last_cmd:
            _last_pose[2] = -70
            pose = _last_pose
        elif '>up' in _last_cmd:
            _last_pose[2] += float(_last_cmd.partition(' ')[2])
            pose = _last_pose
        else:
            pose = _last_pose
        print('update pose of:', self.tello_ip, pose)
        return pose

    def update_cmd(self):
        print('update cmd thread start....')
        pose = self.pose.get()
        self.pose.put(pose)
        if pose[2] == 0:
            while self.Res_flag.value == 0:
                time.sleep(0.1)
            with self.Res_flag.get_lock():
                self.Res_flag.value = 0
            with self.cmd.get_lock():
                self.cmd.value = b'>takeoff'
            self.cmd_event.set()
            print('update cmd, >takeoff')
            time.sleep(0.1)
            self.pose.put(self.update_pos('>takeoff', self.pose.get()))
            while self.Res_flag.value == 0:
                time.sleep(0.1)
            with self.Res_flag.get_lock():
                self.Res_flag.value = 0
            with self.cmd.get_lock():
                self.cmd.value = b'>up 170'
            self.cmd_event.set()
            print('update cmd, >up 170')
            time.sleep(0.1)
            self.pose.put(self.update_pos('>up 170', self.pose.get()))
            while self.Res_flag.value == 0:
                time.sleep(0.1)
            with self.Res_flag.get_lock():
                self.Res_flag.value = 0
            # self.cmd_res.get()
            with self.cmd.get_lock():
                self.cmd.value = b'>streamon'
            self.cmd_event.set()
            print('update cmd, >streamon')
            time.sleep(0.1)
            self.pose.put(self.update_pos('>streamon', self.pose.get()))
        while True:
            # print('in update cmd, main alg {}'.format(self.main_flag.value))
            self.update_path_event.wait()
            pose = self.pose.get()
            self.pose.put(pose)
            # print('path lock acquired by cmd')
            if self.path.empty() is True:
                print('in run thread, path queue is empty, wait for path updating..', self.tello_ip)
                while self.Res_flag.value == 0:
                    time.sleep(0.1)
                with self.Res_flag.get_lock():
                    self.Res_flag.value = 0
                # self.cmd_res.get()
                with self.cmd.get_lock():
                    self.cmd.value = b'wait 5'
                self.cmd_event.set()
                print('update cmd, wait 5')
                time.sleep(5)
                if self.path.empty() is True:
                    print('in run thread, no path updated, break...', self.tello_ip)
                    break
            self.update_path_event.wait()
            target = self.path.get()
            if self.video_flag == 0 and pose[1] < 100:
                if self.main_flag.value == 1:
                    while self.path.empty() is False:
                        self.path.get()
                    break
                while self.Res_flag.value == 0:
                    time.sleep(0.1)
                with self.Res_flag.get_lock():
                    self.Res_flag.value = 0
                # self.cmd_res.get()
                with self.cmd.get_lock():
                    self.cmd.value = b'>streamon'
                self.cmd_event.set()
                print('update cmd, >streamon')
                self.pose.put(self.update_pos('>streamon', self.pose.get()))
                self.video_flag = 1
            self.update_path_event.wait()
            if self.video_flag == 1 and pose[1] >= 100:
                if self.main_flag.value == 1:
                    while self.path.empty() is False:
                        self.path.get()
                    break
                while self.Res_flag.value == 0:
                    time.sleep(0.1)
                with self.Res_flag.get_lock():
                    self.Res_flag.value = 0
                # self.cmd_res.get()
                with self.cmd.get_lock():
                    self.cmd.value = b'>streamoff'
                self.cmd_event.set()
                print('update cmd, >streamoff')
                self.pose.put(self.update_pos('>streamoff', self.pose.get()))
                # self.send_command('>streamoff')
                self.video_flag = 0
            print("--------------------------")
            print("target:{}".format(target))
            self.update_path_event.wait()
            old = time.time()
            self.update_path_event.wait()
            if np.linalg.norm(target[0:3] - pose[0:3]) < 50:
                pass
            else:
                alpha = pose[3] * 3.1416 / 180
                m = np.array([[np.cos(alpha), np.sin(alpha), 0],
                              [-np.sin(alpha), np.cos(alpha), 0],
                              [0, 0, 1]])
                tmp = target[0:3] - pose[0:3]
                tmp = np.dot(m, tmp)
                tmp = np.append(tmp, 100)
                tmp = [int(i) for i in tmp]
                tmp = [str(i) for i in tmp]
                cmd = 'go ' + ' '.join(tmp)
                if self.main_flag.value == 1:
                    while self.path.empty() is False:
                        self.path.get()
                    break
                while self.Res_flag.value == 0:
                    time.sleep(0.1)
                with self.Res_flag.get_lock():
                    self.Res_flag.value = 0
                # self.cmd_res.get()
                with self.cmd.get_lock():
                    self.cmd.value = ('>' + cmd).encode()
                self.cmd_event.set()
                print('update cmd, >' + cmd)
                self.pose.put(self.update_pos('>' + cmd, self.pose.get()))
                # self.send_command('>' + cmd)
            # print('path lock released by cmd')
            if self.main_flag.value == 1:
                while self.path.empty() is False:
                    self.path.get()
                break
            theta = target[3] - pose[3]
            if abs(theta) > 30:
                if abs(theta) > 180:
                    cmd = 'cw ' + str(theta + 180)
                    while self.Res_flag.value == 0:
                        time.sleep(0.1)
                    with self.Res_flag.get_lock():
                        self.Res_flag.value = 0
                    # self.cmd_res.get()
                    with self.cmd.get_lock():
                        self.cmd.value = ('>' + cmd).encode()
                    self.cmd_event.set()
                    print('update cmd, >' + cmd)
                    self.pose.put(self.update_pos('>' + cmd, self.pose.get()))
                    # self.send_command(">" + cmd)
                    cmd = 'ccw ' + str(theta)
                else:
                    cmd = 'ccw ' + str(theta)
                    while self.Res_flag.value == 0:
                        time.sleep(0.1)
                    with self.Res_flag.get_lock():
                        self.Res_flag.value = 0
                    # self.cmd_res.get()
                    with self.cmd.get_lock():
                        self.cmd.value = ('>' + cmd).encode()
                    self.cmd_event.set()
                    print('update cmd, >' + cmd)
                    self.pose.put(self.update_pos('>' + cmd, self.pose.get()))
                    # self.send_command(">" + cmd)
            time.sleep(0.1)
        while self.Res_flag.value == 0:
            time.sleep(0.1)
            if self.main_flag.value == 1:
                break
        with self.Res_flag.get_lock():
            self.Res_flag.value = 0
        self.send_command(">land")
        while self.Res_flag.value == 0:
            time.sleep(0.1)
            if self.main_flag.value == 1:
                break
        with self.Res_flag.get_lock():
            self.Res_flag.value = 0
        self.send_command(">streamoff")
        print('run dying...', self.tello_ip)
        time.sleep(2)
        self.run_thread_flag.value = 1
        print('update thread died...')

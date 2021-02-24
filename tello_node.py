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
import face_recognition
import psutil
import os
import av
import av.datasets
from io import BytesIO
import signal
# from memory_profiler import profile
# from guppy import hpy
# import gc

class TelloNode:
    def __init__(self, tello_info, res_flag, main_flag, p_flag, show_match_flag):
        self.tello_ip = tello_info[0]
        self.ctr_port = tello_info[1]
        self.video_port = tello_info[2]
        self.ctr_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.ctr_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.ctr_socket.bind(('', self.ctr_port))
        self.pose = multiprocessing.Queue()
        self.video_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.video_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 10 * 1024)
        self.video_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        # self.video_socket.settimeout(0.5)
        self.video_socket.bind(('', self.video_port))
        self.h264decoder = h264decoder.H264Decoder()
        self.queue = multiprocessing.Queue(2)
        self.queue_up_camera = multiprocessing.Queue()
        self.pose_estimater = PoseEstimater('SIFT', 25)
        self.pose_estimater.loaddata('pose_estimater/dataset/')
        if show_match_flag == 1:
            self.pose_estimater.show_match_start()
        self.path = multiprocessing.Queue()
        self.video_flag = 1
        self.run_thread_flag = multiprocessing.Value('i', 0)
        self.path_empty_flag = multiprocessing.Value('i', 0)
        self.update_path_flag = multiprocessing.Value('i', 0)
        self.takeoff_flag = multiprocessing.Value('i', 1)
        self.Res_flag = res_flag
        self.cmd = multiprocessing.Array('c', 30)
        self.cmd_event = multiprocessing.Event()
        self.update_path_event = multiprocessing.Event()
        self.main_flag = main_flag
        self.permission_flag = p_flag
        self.target = multiprocessing.Array('c', 40)
        self.init_pose = []
        self.show_video_thread = multiprocessing.Process(target=self.show_pic)
        self.show_video_thread.start()
        self.switch_count = 3
        self.thread_list = []
        self.scan_face_flag = multiprocessing.Value('i', 0)
        self.face_point = multiprocessing.Array('c', 30)
        self.face_detected_pose = multiprocessing.Array('c', 40)

    def scan_face(self):
        while True:
            if self.queue_up_camera.empty() is False:
                img = self.queue_up_camera.get()
                img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
                time1 = time.time()
                pose = self.pose.get()
                self.pose.put(pose)
                for i in range(len(pose)):
                    pose[i] = int(pose[i])
                self.face_detected_pose.value = ','.join(map(str, pose)).encode()
                locate = face_recognition.face_locations(img, number_of_times_to_upsample=2, model='hog')
                print("face recognition needs {}s.".format(time.time()-time1))
                print(locate)
                try:
                    if len(locate) != 0:
                        # c_point = np.array([int(locate[0][0]/2+locate[0][2]/2), int(locate[0][1]/2+locate[0][3]/2)])
                        self.face_point.value = ','.join(map(str, locate[0])).encode()
                    else:
                        self.face_point.value = ''.encode()
                except ValueError as e:
                    self.face_point.value = ''.encode()
            time.sleep(1)

    def get_up_camera_image(self):
        if self.queue_up_camera.empty() is True:
            return None
        else:
            _dict = {}
            pose = self.pose.get()
            self.pose.put(pose)
            _dict['image'] = self.queue_up_camera.get()
            _dict['pose'] = pose
            return _dict

    def show_pic(self):
        # frame_dict = {}.fromkeys(self.tello_ip_list,bytes())
        print("show pic thread start.\n")
        while True:
            if self.main_flag.value == 1:
                while self.path.empty() is False:
                    self.path.get()
                break
            a = cv.waitKey(2)
            if a == ord('q'):
                cv.destroyAllWindows()
                break
            '''for tello in self.tello_list:
                flag.append(self.queue[tello.tello_ip].empty())
            tmp = set(flag)
            if (flag[0] == 0) and (len(tmp) == 1):
                for tello in self.tello_list:
                    f[tello.tello_ip] = self.queue[tello.tello_ip].get()
                    cv2.imshow(tello.tello_ip, f[tello.tello_ip])
            flag = []'''
            if self.queue.empty() is False:
                f = self.queue.get()
                if self.queue.empty():
                    self.queue.put(f)
                cv.imshow(self.tello_ip, f)
            time.sleep(0.02)
            # if self.queue_up_camera.empty() is False:
            #     f = self.queue_up_camera.get()
            #     self.queue_up_camera.put(f)
            #     cv.imshow(self.tello_ip+'-up-camera', f)
            #     time.sleep(0.01)

    def get_target(self):
        if self.target.value.decode() == '':
            return None
        else:
            tmp = self.target.value
            tmp = tmp.decode().split(',')
            try:
                tmp = list(map(int, tmp))
            except ValueError as e:
                return None
            return tmp

    def get_face_point(self):
        if self.face_point.value.decode() == '':
            return None, None
        else:
            tmp = self.face_point.value
            tmp = tmp.decode().split(',')
            tmp = list(map(int, tmp))
            tmp2 = self.face_detected_pose.value
            tmp2 = tmp2.decode().split(',')
            tmp2 = list(map(float, tmp2))
            self.face_point.value = ''.encode()
            return tmp, tmp2

    def get_thread_flag(self):
        return self.run_thread_flag

    def init_path(self, _path, pose):
        path = np.array(_path)
        path_x = path[:, 0]
        path_y = path[:, 1]
        path_z = path[:, 2]
        path_theta = path[:, 3]
        for i in range(len(path_x)):
            path_x[i] = int(path_x[i])
            path_y[i] = int(path_y[i])
            path_z[i] = int(path_z[i])
            path_theta[i] = int(path_theta[i])
        for i in range(5):
            path_x = path_x.repeat(2)[:-1]
            path_y = path_y.repeat(2)[:-1]
            path_z = path_z.repeat(2)[:-1]
            path_theta = path_theta.repeat(2)[:-1]
            path_x[1::2] = (path_x[2::2] - path_x[1::2]) / 2 + path_x[1::2]
            path_y[1::2] = (path_y[2::2] - path_y[1::2]) / 2 + path_y[1::2]
            path_z[1::2] = (path_z[2::2] - path_z[1::2]) / 2 + path_z[1::2]
            # path_theta[1::2] = (path_theta[2::2] - path_theta[1::2]) / 2 + path_theta[1::2]
        distance = np.sqrt(np.ediff1d(path_x) ** 2 + np.ediff1d(path_y) ** 2 + np.ediff1d(path_z) ** 2)
        d = 0.0
        equdist_waypoint = [[path_x[0]], [path_y[0]], [path_z[0]], [path_theta[0]]]
        for i in range(len(distance)):
            d = distance[i] + d
            if d >= 50:
                equdist_waypoint = np.append(equdist_waypoint,
                                             [[path_x[i]], [path_y[i]], [path_z[i]], [path_theta[i]]])
                d = 0.0
        equdist_waypoint = np.append(equdist_waypoint, path[-1])
        path = equdist_waypoint.reshape((-1, 4))
        self.update_path_event.clear()
        # print('updating the path______________________________________', self.tello_ip)
        tmp = np.array(path)
        tmp[-2][3] = tmp[-1][3]
        d = np.array([])
        for t in tmp:
            d = np.append(d, np.linalg.norm(np.array(pose[0:3]) - t[0:3], 2))
        a = np.argmin(d)
        while self.path.empty() is False:
            self.path.get()
        if a >= len(tmp) - 2:
            for t in tmp:
                self.path.put(t)
        else:
            for t in tmp[a + 1:]:
                self.path.put(t)
        self.pose.put(pose)
        self.target.value = ','.join(map(str, pose)).encode()
        self.init_pose = pose

    def update_path(self, path):
        update_path_thread = multiprocessing.Process(target=self._update_path, args=(path,))
        update_path_thread.start()

    def _update_path(self, path1):
        if self.takeoff_flag.value == 1:
            pass
        else:
            pose = self.pose.get()
            self.pose.put(pose)
            last_path = self.path.get()
            # self.path.put(last_path)
            pose[3] = last_path[3]
            path = np.array(path1)
            path = np.insert(path, 0, pose, axis=0)
            path_x = path[:, 0]
            path_y = path[:, 1]
            path_z = path[:, 2]
            path_theta = path[:, 3]
            for i in range(len(path_x)):
                path_x[i] = int(path_x[i])
                path_y[i] = int(path_y[i])
                path_z[i] = int(path_z[i])
                path_theta[i] = int(path_theta[i])
            for i in range(5):
                path_x = path_x.repeat(2)[:-1]
                path_y = path_y.repeat(2)[:-1]
                path_z = path_z.repeat(2)[:-1]
                path_theta = path_theta.repeat(2)[:-1]
                path_x[1::2] = (path_x[2::2] - path_x[1::2]) / 2 + path_x[1::2]
                path_y[1::2] = (path_y[2::2] - path_y[1::2]) / 2 + path_y[1::2]
                path_z[1::2] = (path_z[2::2] - path_z[1::2]) / 2 + path_z[1::2]
                # path_theta[1::2] = (path_theta[2::2] - path_theta[1::2]) / 2 + path_theta[1::2]
            distance = np.sqrt(np.ediff1d(path_x) ** 2 + np.ediff1d(path_y) ** 2 + np.ediff1d(path_z) ** 2)
            d = 0.0
            equdist_waypoint = [[path_x[0]], [path_y[0]], [path_z[0]], [path_theta[0]]]
            for i in range(len(distance)):
                d = distance[i] + d
                if d >= 50:
                    equdist_waypoint = np.append(equdist_waypoint,
                                                 [[path_x[i]], [path_y[i]], [path_z[i]], [path_theta[i]]])
                    d = 0.0
            equdist_waypoint = np.append(equdist_waypoint, path[-1])
            path = equdist_waypoint.reshape((-1, 4))
            self.update_path_event.clear()
            print('updating the path______________________________________', self.tello_ip)
            tmp = np.array(path)
            tmp[-2][3] = tmp[-1][3]
            print("update path:", tmp)
            d = np.array([])
            for t in tmp:
                d = np.append(d, np.linalg.norm(np.array(pose[0:3]) - t[0:3], 2))
            a = np.argmin(d)
            while self.path.empty() is False:
                self.path.get()
            if a >= len(tmp)-2:
                for t in tmp:
                    self.path.put(t)
            else:
                for t in tmp[a + 1:]:
                    self.path.put(t)
            print('finished updating the path_________________________________', self.tello_ip)
            self.update_path_event.set()

    def _h264_decode(self, packet_data, queue):
        frames = self.h264decoder.decode(packet_data)
        # packet = av.Packet(packet_data)
        # print(type(packet))
        # frames = av.packet.Packet.decode(packet)
        # print(frames)
        for frame_data in frames:
            (frame, w, h, ls) = frame_data
            if frame is not None:
                frame = np.fromstring(frame, dtype=np.ubyte, count=len(frame), sep='')
                frame = (frame.reshape((h, ls // 3, 3)))
                frame = frame[:, :w, :]
                # if h == 240:
                while queue.qsize() > 1:
                    queue.get()
                queue.put(frame)

    def _receive_video_thread(self, queue):
        pack_data = ''
        # buffer = BytesIO()
        print("receive video thread start....")
        while True:
            try:
                # print("in the receive video thread while loop...")
                res_string, ip = self.video_socket.recvfrom(2048)
                # buffer.write(res_string)
                pack_data += res_string.hex()
                if len(res_string) != 1460:
                    # print("The size of packet data is %d.\n" % len(pack_data))
                    tmp = bytes.fromhex(pack_data)
                    # buffer.seek(0)
                    self._h264_decode(tmp, queue)
                    # self.Queue_res_buf.put(self.res_string)
                    pack_data = ''
                    # buffer.seek(0)
                    # p = av.open(buffer)
                    # print(type(p))
                    # print(p)
                    # print(len(buffer.getvalue()))
                    # print(buffer.getvalue())
                    # container = av.open(buffer, 'r')
                    # print(container)
                    # frames = container.decode()
                    # print(frames)
            except socket.error as exc:
                print("Caught exception socket.error(video_thread): %s" % exc)

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
        video_thread = multiprocessing.Process(target=self._receive_video_thread, args=(self.queue,))
        video_thread.start()
        os.system("taskset -cp 12,15 " + str(video_thread.pid))
        print("video pid of {} is {}".format(self.tello_ip, video_thread.pid))
        cmd_thread = multiprocessing.Process(target=self.update_cmd)
        cmd_thread.start()
        print("cmd thread of {} is {}".format(self.tello_ip, cmd_thread.pid))
        time.sleep(1)
        scan_face_thread = multiprocessing.Process(target=self.scan_face)
        scan_face_thread.start()
        print("scan_face thread of {} is {}".format(self.tello_ip, scan_face_thread.pid))
        old_time = time.time()
        with self.cmd.get_lock():
            self.cmd.value = b'>command'
        self.cmd_event.set()
        self.update_path_event.set()
        while True:
            # if time.time() - old_time > 10 and self.takeoff_flag.value == 0:
            #     # self.thread_list = psutil.pids()
            #     # if video_thread.pid not in self.thread_list:
            #     #     print("video thread pid not in the list, restart thread.")
            #     #     video_thread.start()
            #     #     print("new video pid of {} is {}".format(self.tello_ip, video_thread.pid))
            #     # else:
            #     #     print("video thread pid {} in the list".format(video_thread.pid))
            #     # test_list = []
            #     # for i in range(10):
            #     #     p_cpu = p.cpu_percent(interval=0.1)
            #     #     test_list.append(p_cpu)
            #     # per = sum(test_list)/len(test_list)
            #     # if last_per == 0 and per == 0:
            #     #     print("Kill the video thread.")
            #     #     os.kill(video_thread.pid, signal.SIGKILL)
            #     #     print("restart video thread.")
            #     #     video_thread = multiprocessing.Process(target=self._receive_video_thread)
            #     #     video_thread.start()
            #     #     print("new video thread pid:", video_thread.pid)
            #     #     last_per = 1
            #     # else:
            #     #     last_per = per
            #     # print("Process {}: cpu percent:{}".format(video_thread.pid, sum(test_list)/len(test_list)))
            #     p = psutil.Process(video_thread.pid)
            #     info = p.memory_full_info()
            #     memory = info.uss/1024./1024./1024.
            #     print("memory used by {}:{}g".format(self.tello_ip, memory))
            #     if memory > 3:
            #         for child in p.children(recursive=True):
            #             child.kill()
            #         p.kill()  ## this program
            #         video_thread = multiprocessing.Process(target=self._receive_video_thread, args=(self.queue,))
            #         video_thread.start()
            #         os.system("taskset -cp 12,15 " + str(video_thread.pid))
            #         print("new video pid of {} is {}".format(self.tello_ip, video_thread.pid))
            #     old_time = time.time()
            if self.run_thread_flag.value == 1 or self.main_flag.value == 1:
                time.sleep(1)
                break
            time.sleep(0.1)
            self.cmd_event.wait()
            with self.cmd.get_lock():
                cmd = self.cmd.value.decode()
            self.send_command(cmd)
            if 'm' not in cmd:
                print('in run thread, send ', cmd)
            self.cmd_event.clear()
            # if cmd_thread.is_alive() is False:
            #     print('update_cmd thread is died...')
        print('run thread died...')

    def run(self):
        run_thread = multiprocessing.Process(target=self.run_thread)
        run_thread.start()
        print("run pid of {} is {}".format(self.tello_ip, run_thread.pid))
        # last_per = 1
        print('run thread start....')

    def update_pos(self, _last_cmd, _last_pose):
        time.sleep(0.1)
        # img = None
        pose = np.array([0, 0, 0, 0])
        if '>ccw' in _last_cmd:
            angle = float(_last_cmd.partition(' ')[2])
            # print('angle:{}'.format(angle))
            pose[3] = angle + _last_pose[3]
            if pose[3] >= 360:
                pose[3] -= 360
            pose[0:3] = _last_pose[0:3]
        elif '>cw' in _last_cmd:
            angle = float(_last_cmd.partition(' ')[2])
            # print('angle:{}'.format(angle))
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
            _last_pose[2] = 100
            pose = _last_pose
        elif '>up' in _last_cmd:
            _last_pose[2] += float(_last_cmd.partition(' ')[2])
            pose = _last_pose
        else:
            pose = _last_pose
        if self.queue.empty() is False:
            img = self.queue.get()
            # self.queue.put(img)
            if img is not None:
                _pose, yaw = self.pose_estimater.estimate_pose(img, pose)
                # if _pose is not None:
                #     print('in img1')
                #     while self.queue.empty() is True:
                #         time.sleep(0.01)
                #     img = self.queue.get()
                #     _pose, yaw = self.pose_estimater.estimate_pose(img)
                if _pose is not None:
                    _pose = _pose.reshape(1, -1)[0]
                    if np.linalg.norm(pose[0:2]-_pose[0:2]) < 150:
                        print('in img2')
                        pose[0] = int(_pose[0])
                        pose[1] = int(_pose[1])
                        # if _pose[2] == 0:
                        #     pose[2] = _last_pose[2]
                        # else:
                        #     pose[2] = _pose[2]
                        pose[2] = int(_pose[2])
                        if yaw < 0:
                            yaw += 360
                        pose[3] = int(yaw)
                        # print('update pose of:', self.tello_ip, pose)
                        # return pose
                    else:
                        print("error:", np.linalg.norm(pose[0:2]-_pose[0:2]))
                        print("_pose: ", _pose)
        print('update pose of:', self.tello_ip, pose)
        return pose

    def get_path_status(self):
        return self.path_empty_flag.value

    def update_cmd(self):
        print('update cmd thread start....')
        pose = self.pose.get()
        self.pose.put(pose)
        old_time = time.time()
        cmd = ''
        camera_flag = 1
        count = 0
        # time2 = time.time()
        if pose[2] == 85:
            self.target.value = ','.join(map(str, self.init_pose)).encode()
            time.sleep(0.1)
            while self.permission_flag.value == 0:
                # print('wait for permission of ', self.tello_ip, self.permission_flag.value)
                if time.time() - old_time >= 5:
                    with self.cmd.get_lock():
                        self.cmd.value = b'>command'
                        print('update cmd, >command')
                    self.cmd_event.set()
                    old_time = time.time()
                time.sleep(0.1)
            self.permission_flag.value = 0
            while self.Res_flag.value == 0:
                time.sleep(0.1)
            with self.Res_flag.get_lock():
                self.Res_flag.value = 0
            # self.cmd_res.get()
            self.pose.put(self.update_pos('>streamon', self.pose.get()))
            with self.cmd.get_lock():
                self.cmd.value = b'>streamon'
            self.cmd_event.set()
            while self.Res_flag.value == 0:
                time.sleep(0.1)
            with self.Res_flag.get_lock():
                self.Res_flag.value = 0
            with self.cmd.get_lock():
                self.cmd.value = b'>takeoff'
            self.cmd_event.set()
            # print('update cmd, >takeoff')
            self.pose.put(self.update_pos('>takeoff', self.pose.get()))
            tmp = self.pose.get()
            self.pose.put(tmp)
            self.target.value = ','.join(map(str, tmp)).encode()
            # while self.permission_flag.value == 0:
            #     # print('wait for permission 1.5,', self.permission_flag.value)
            #     old_time = time.time()
            #     # print('wait for permission of ', self.tello_ip, self.permission_flag.value)
            #     if time.time() - old_time >= 5:
            #         with self.cmd.get_lock():
            #             self.cmd.value = b'>command'
            #         self.cmd_event.set()
            #     time.sleep(0.2)
            # self.permission_flag.value = 0
            while self.Res_flag.value == 0:
                time.sleep(0.1)
            with self.Res_flag.get_lock():
                self.Res_flag.value = 0
            self.pose.put(self.update_pos('>up 130', self.pose.get()))
            with self.cmd.get_lock():
                self.cmd.value = b'>up 100'
            self.cmd_event.set()
            # print('update cmd, >up 130')
            while self.Res_flag.value == 0:
                time.sleep(0.1)
            with self.Res_flag.get_lock():
                self.Res_flag.value = 0
            # self.cmd_res.get()
            self.pose.put(self.update_pos('>streamon', self.pose.get()))
            with self.cmd.get_lock():
                self.cmd.value = b'>command'
            self.cmd_event.set()
            # old_time = time.time()
            # while self.Res_flag.value == 0:
            #     if time.time() - old_time > 5:
            #         with self.cmd.get_lock():
            #             self.cmd.value = ('>' + cmd).encode()
            #         self.cmd_event.set()
            #         old_time = time.time()
            #     time.sleep(0.1)
            # with self.Res_flag.get_lock():
            #     self.Res_flag.value = 0
            # # self.cmd_res.get()
            # self.pose.put(self.update_pos('>setersolution low', self.pose.get()))
            # with self.cmd.get_lock():
            #     self.cmd.value = b'>setresolution low'
            # self.cmd_event.set()
            # print('update cmd, >setresolution low')
            # while self.Res_flag.value == 0:
            #     time.sleep(0.1)
            # with self.Res_flag.get_lock():
            #     self.Res_flag.value = 0
            # # self.cmd_res.get()
            # self.pose.put(self.update_pos('>setfps high', self.pose.get()))
            # with self.cmd.get_lock():
            #     self.cmd.value = b'>setfps high'
            # self.cmd_event.set()
            # # print('update cmd, >setfps high')
            self.takeoff_flag.value = 0
        while True:
            # print('in update cmd, main alg {}'.format(self.main_flag.value))
            # print('in update cmd, permission {}'.format(self.permission_flag.value))
            # self.update_path_event.wait()
            # pose = self.pose.get()
            # self.pose.put(pose)
            # print('path lock acquired by cmd')
            # if self.path.empty() is True:
            #     print('in run thread, path queue is empty, wait for path updating..', self.tello_ip)
            #     while self.Res_flag.value == 0:
            #         time.sleep(0.1)
            #     with self.Res_flag.get_lock():
            #         self.Res_flag.value = 0
            #     # self.cmd_res.get()
            #     with self.cmd.get_lock():
            #         self.cmd.value = b'wait 10'
            #     self.cmd_event.set()
            #     print('update cmd, wait 10')
            #     time.sleep(10)
            #     if self.path.empty() is True:
            #         print('in run thread, no path updated, break...', self.tello_ip)
            #         break
            self.update_path_event.wait()
            # count = count + 1
            # time.sleep(0.1)
            old_time = time.time()
            while self.permission_flag.value == 0:
                # print('wait for permission, 2', self.permission_flag.value)
                # print('wait for permission of ', self.tello_ip, self.permission_flag.value)
                if time.time() - old_time >= 5:
                    with self.cmd.get_lock():
                        self.cmd.value = b'>command'
                        # print('update cmd, >command')
                    self.cmd_event.set()
                    old_time = time.time()
                time.sleep(0.2)
            self.permission_flag.value = 0
            old_time = time.time()
            while self.Res_flag.value == 0:
                if time.time() - old_time > 5:
                    with self.cmd.get_lock():
                        self.cmd.value = ('>command').encode()
                    self.cmd_event.set()
                    old_time = time.time()
                time.sleep(0.1)
            with self.Res_flag.get_lock():
                self.Res_flag.value = 0
            self.pose.put(self.update_pos('>' + cmd, self.pose.get()))
            pose = self.pose.get()
            self.pose.put(pose)
            target = self.path.get()
            if self.path.empty() is True:
                self.path_empty_flag.value = 1
                self.path.put(target)
                time.sleep(1)
            else:
                self.path_empty_flag.value = 0
            self.target.value = ','.join(map(str, target)).encode()
            print("--------------------------")
            print("target:{}".format(target))
            if np.linalg.norm(target[0:3] - pose[0:3]) < 50:
                cmd = 'command'
                with self.cmd.get_lock():
                    self.cmd.value = ('>' + cmd).encode()
                self.cmd_event.set()
            else:
                alpha = pose[3] * 3.1416 / 180
                m = np.array([[np.cos(alpha), np.sin(alpha), 0],
                              [-np.sin(alpha), np.cos(alpha), 0],
                              [0, 0, 1]])
                tmp = target[0:3] - pose[0:3]
                tmp = np.dot(m, tmp)
                if abs(tmp[0]) + abs(tmp[1]) > 50:
                    if tmp[2] > 15:
                        tmp[2] = 15
                    elif tmp[2] < -15:
                        tmp[2] = -15
                tmp = np.append(tmp, 100)
                tmp = [int(i) for i in tmp]
                tmp = [str(i) for i in tmp]
                cmd = 'go ' + ' '.join(tmp)
                if self.main_flag.value == 1:
                    while self.path.empty() is False:
                        self.path.get()
                    break
                # self.cmd_res.get()
                try:
                    with self.cmd.get_lock():
                        self.cmd.value = ('>' + cmd).encode()
                    self.cmd_event.set()
                except ValueError:
                    print(cmd)
                    with self.cmd.get_lock():
                        self.cmd.value = b'>command'
                    self.cmd_event.set()
                # print('update cmd, >' + cmd)
                # self.send_command('>' + cmd)
            # print('path lock released by cmd')
            if self.main_flag.value == 1:
                while self.path.empty() is False:
                    self.path.get()
                break
            old_time = time.time()
            while self.Res_flag.value == 0:
                if time.time() - old_time > 5:
                    with self.cmd.get_lock():
                        self.cmd.value = ('>' + cmd).encode()
                    self.cmd_event.set()
                    old_time = time.time()
                time.sleep(0.1)
            with self.Res_flag.get_lock():
                self.Res_flag.value = 0
            self.pose.put(self.update_pos('>' + cmd, self.pose.get()))
            pose = self.pose.get()
            self.pose.put(pose)
            # if camera_flag == 1 and count == self.switch_count:
            #     with self.cmd.get_lock():
            #         self.cmd.value = b'>downvision 1'
            #     self.cmd_event.set()
            #     camera_flag = 0
            #     count = 0
            #     time.sleep(0.2)
            # if camera_flag == 0 and count == self.switch_count:
            #     with self.cmd.get_lock():
            #         self.cmd.value = b'>downvision 0'
            #     self.cmd_event.set()
            #     camera_flag = 1
            #     count = 0
            #     time.sleep(0.2)
            # else:
            #     cmd = 'command'
            #     with self.cmd.get_lock():
            #         self.cmd.value = ('>' + cmd).encode()
            #     self.cmd_event.set()
            #     time.sleep(0.1)
            # old_time = time.time()
            # while self.Res_flag.value == 0:
            #     if time.time() - old_time > 5:
            #         with self.cmd.get_lock():
            #             self.cmd.value = ('>' + cmd).encode()
            #         self.cmd_event.set()
            #         old_time = time.time()
            #     time.sleep(0.1)
            # with self.Res_flag.get_lock():
            #     self.Res_flag.value = 0
            theta = target[3] - pose[3]
            if abs(theta) > 5:
                if theta < 0:
                    if abs(theta) > 180:
                        cmd = 'ccw ' + str(theta + 360)
                        # self.cmd_res.get()
                        with self.cmd.get_lock():
                            self.cmd.value = ('>' + cmd).encode()
                        self.cmd_event.set()
                        # print('update cmd, >' + cmd)
                        # self.send_command(">" + cmd)
                    else:
                        cmd = 'ccw ' + str(theta)
                        with self.cmd.get_lock():
                            self.cmd.value = ('>' + cmd).encode()
                        self.cmd_event.set()
                        # print('update cmd, >' + cmd)
                        # self.send_command(">" + cmd)
                else:
                    if abs(theta) > 180:
                        cmd = 'ccw ' + str(theta - 360)
                        # self.cmd_res.get()
                        with self.cmd.get_lock():
                            self.cmd.value = ('>' + cmd).encode()
                        self.cmd_event.set()
                        # print('update cmd, >' + cmd)
                        # self.send_command(">" + cmd)
                    else:
                        cmd = 'ccw ' + str(theta)
                        with self.cmd.get_lock():
                            self.cmd.value = ('>' + cmd).encode()
                        self.cmd_event.set()
                        # print('update cmd, >' + cmd)
                        # self.send_command(">" + cmd)
            else:
                cmd = 'command'
                with self.cmd.get_lock():
                    self.cmd.value = ('>' + cmd).encode()
                self.cmd_event.set()
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

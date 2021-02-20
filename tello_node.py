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
import psutil


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
        self.video_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 512 * 1024)
        self.video_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.video_socket.bind(('', self.video_port))
        self.h264decoder = h264decoder.H264Decoder()
        self.queue = multiprocessing.Queue()
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
        self.cmd = multiprocessing.Array('c', 20)
        self.cmd_event = multiprocessing.Event()
        self.update_path_event = multiprocessing.Event()
        self.main_flag = main_flag
        self.permission_flag = p_flag
        self.target = multiprocessing.Array('c', 20)
        self.init_pose = []
        self.show_video_thread = multiprocessing.Process(target=self.show_pic)
        self.show_video_thread.start()
        self.switch_count = 2
        self.thread_list = []

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
                self.queue.put(f)
                cv.imshow(self.tello_ip, f)
                time.sleep(0.01)
            if self.queue_up_camera.empty() is False:
                f = self.queue_up_camera.get()
                self.queue_up_camera.put(f)
                cv.imshow(self.tello_ip+'-up-camera', f)
                time.sleep(0.01)

    def get_target(self):
        if self.target.value.decode() == '':
            return None
        else:
            tmp = self.target.value
            tmp = tmp.decode().split(',')
            tmp = list(map(int, tmp))
            return tmp

    def get_thread_flag(self):
        return self.run_thread_flag

    def init_path(self, _path, pose):
        path = np.array(_path)
        path_x = path[:, 0]
        path_y = path[:, 1]
        path_z = path[:, 2]
        path_theta = path[:, 3]
        for i in range(5):
            path_x = path_x.repeat(2)[:-1]
            path_y = path_y.repeat(2)[:-1]
            path_z = path_z.repeat(2)[:-1]
            path_theta = path_theta.repeat(2)[:-1]
            path_x[1::2] = (path_x[2::2] - path_x[1::2]) / 2 + path_x[1::2]
            path_y[1::2] = (path_y[2::2] - path_y[1::2]) / 2 + path_y[1::2]
            path_z[1::2] = (path_z[2::2] - path_z[1::2]) / 2 + path_z[1::2]
            path_theta[1::2] = (path_theta[2::2] - path_theta[1::2]) / 2 + path_theta[1::2]
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
        for t in path:
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
            path = np.array(path1)
            path_x = path[:, 0]
            path_y = path[:, 1]
            path_z = path[:, 2]
            path_theta = path[:, 3]
            for i in range(5):
                path_x = path_x.repeat(2)[:-1]
                path_y = path_y.repeat(2)[:-1]
                path_z = path_z.repeat(2)[:-1]
                path_theta = path_theta.repeat(2)[:-1]
                path_x[1::2] = (path_x[2::2] - path_x[1::2]) / 2 + path_x[1::2]
                path_y[1::2] = (path_y[2::2] - path_y[1::2]) / 2 + path_y[1::2]
                path_z[1::2] = (path_z[2::2] - path_z[1::2]) / 2 + path_z[1::2]
                path_theta[1::2] = (path_theta[2::2] - path_theta[1::2]) / 2 + path_theta[1::2]
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
            d = np.array([])
            pose = self.pose.get()
            self.pose.put(pose)
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

    def _h264_decode(self, packet_data):
        frames = self.h264decoder.decode(packet_data)
        for frame_data in frames:
            (frame, w, h, ls) = frame_data
            if frame is not None:
                frame = np.fromstring(frame, dtype=np.ubyte, count=len(frame), sep='')
                frame = (frame.reshape((h, ls // 3, 3)))
                frame = frame[:, :w, :]
                if h == 240 and ls == 960:
                    while self.queue.empty() is False:
                        self.queue.get()
                    self.queue.put(frame)
                else:
                    while self.queue_up_camera.empty() is False:
                        self.queue_up_camera.get()
                    self.queue_up_camera.put(frame)

    def _receive_video_thread(self):
        pack_data = ''
        print("receive video thread start....")
        oldtime = time.time()
        while True:
            res_string, ip = self.video_socket.recvfrom(2000)
            pack_data += res_string.hex()
            if len(res_string) != 1460:
                try:
                    self._h264_decode(bytes.fromhex(pack_data))
                except:
                    break
                pack_data = ''
                # if time.time() - oldtime > 5:
                #     print("get video data from", self.tello_ip)
                #     oldtime = time.time()
            if self.run_thread_flag.value == 1 or self.main_flag.value == 1:
                time.sleep(1)
                # self.video_socket.close()
                print('video thread die..')
                break
            # except:
            #     print("Caught exception socket.error(video_thread): %s in node" % exc)
            #     break
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
        print("video pid of {} is {}".format(self.tello_ip, video_thread.pid))
        cmd_thread = multiprocessing.Process(target=self.update_cmd, daemon=True)
        cmd_thread.start()
        print("cmd thread of {} is {}".format(self.tello_ip, cmd_thread.pid))
        time.sleep(1)
        old_time = time.time()
        with self.cmd.get_lock():
            self.cmd.value = b'>command'
        self.cmd_event.set()
        self.update_path_event.set()
        while True:
            if time.time() - old_time > 5:
                self.thread_list = psutil.pids()
                print(self.thread_list)
                if video_thread.pid not in self.thread_list:
                    print("video thread pid not in the list, restart thread.")
                    video_thread.start()
                    print("new video pid of {} is {}".format(self.tello_ip, video_thread.pid))
                else:
                    print("video thread pid {} in the list".format(video_thread.pid))
                old_time = time.time()
            # print('in run , target:', self.target.value)
            # print("video thread is alive,", video_thread.is_alive())
            # print("cmd thread is alive,", cmd_thread.is_alive())
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

    def update_pos(self, _last_cmd, _last_pose):
        time.sleep(0.3)
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
                    # print('in img2')
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
                self.cmd.value = b'>up 130'
            self.cmd_event.set()
            # print('update cmd, >up 130')
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
            # self.cmd_res.get()
            self.pose.put(self.update_pos('>setersolution low', self.pose.get()))
            with self.cmd.get_lock():
                self.cmd.value = b'>setresolution low'
            self.cmd_event.set()
            # print('update cmd, >setresolution low')
            while self.Res_flag.value == 0:
                time.sleep(0.1)
            with self.Res_flag.get_lock():
                self.Res_flag.value = 0
            # self.cmd_res.get()
            self.pose.put(self.update_pos('>setfps high', self.pose.get()))
            with self.cmd.get_lock():
                self.cmd.value = b'>setfps high'
            self.cmd_event.set()
            # print('update cmd, >setfps high')
            while self.Res_flag.value == 0:
                time.sleep(0.1)
            with self.Res_flag.get_lock():
                self.Res_flag.value = 0
            # self.cmd_res.get()
            self.pose.put(self.update_pos('>downvision 1', self.pose.get()))
            with self.cmd.get_lock():
                self.cmd.value = b'>downvision 1'
            self.cmd_event.set()
            # print('update cmd, >downvision 1')
            self.takeoff_flag.value = 0
        while True:
            # print('in update cmd, main alg {}'.format(self.main_flag.value))
            # print('in update cmd, permission {}'.format(self.permission_flag.value))
            self.update_path_event.wait()
            pose = self.pose.get()
            self.pose.put(pose)
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
            count = count + 1
            # time.sleep(0.1)
            while self.permission_flag.value == 0:
                # print('wait for permission, 2', self.permission_flag.value)
                old_time = time.time()
                # print('wait for permission of ', self.tello_ip, self.permission_flag.value)
                if time.time() - old_time >= 5:
                    with self.cmd.get_lock():
                        self.cmd.value = b'>command'
                        # print('update cmd, >command')
                    self.cmd_event.set()
                time.sleep(0.2)
            self.permission_flag.value = 0
            while self.Res_flag.value == 0:
                time.sleep(0.1)
            with self.Res_flag.get_lock():
                self.Res_flag.value = 0
            # with self.cmd.get_lock():
            #     self.cmd.value = b'>downvision 1'
            # self.cmd_event.set()
            self.pose.put(self.update_pos('>' + cmd, self.pose.get()))
            # with self.cmd.get_lock():
            #     self.cmd.value = b'>downvision 0'
            # self.cmd_event.set()
            pose = self.pose.get()
            self.pose.put(pose)
            if np.linalg.norm(target[0:3] - pose[0:3]) < 25:
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
                if tmp[2] > 10:
                    tmp[2] = 10
                elif tmp[2] < -10:
                    tmp[2] = -10
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
                # print('update cmd, >' + cmd)
                # self.send_command('>' + cmd)
            # print('path lock released by cmd')
            if self.main_flag.value == 1:
                while self.path.empty() is False:
                    self.path.get()
                break
            while self.Res_flag.value == 0:
                time.sleep(0.1)
            with self.Res_flag.get_lock():
                self.Res_flag.value = 0
            self.pose.put(self.update_pos('>' + cmd, self.pose.get()))
            pose = self.pose.get()
            self.pose.put(pose)
            if camera_flag == 1 and count == self.switch_count:
                with self.cmd.get_lock():
                    self.cmd.value = b'>downvision 1'
                self.cmd_event.set()
                camera_flag = 0
                count = 0
                time.sleep(0.2)
            if camera_flag == 0 and count == self.switch_count:
                with self.cmd.get_lock():
                    self.cmd.value = b'>downvision 0'
                self.cmd_event.set()
                camera_flag = 1
                count = 0
                time.sleep(0.2)
            else:
                with self.cmd.get_lock():
                    self.cmd.value = b'>command'
                self.cmd_event.set()
            while self.Res_flag.value == 0:
                time.sleep(0.1)
            with self.Res_flag.get_lock():
                self.Res_flag.value = 0
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

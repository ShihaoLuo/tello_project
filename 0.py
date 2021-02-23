# -*- coding: utf-8 -*-
# @Time    : 2021/1/1 下午4:50
# @Author  : JakeShihao Luo
# @Email   : jakeshihaoluo@gmail.com
# @File    : 0.py
# @Software: PyCharm

from scanner import *
import socket
from tello_node import *
import multiprocessing
import numpy as np
import copy


def scheduler(tello_node, permission_flag):
    print('scheduler thread start...')
    while True:
        permission_list = []
        ban_list = []
        target = {}
        tmp = []
        candidate_list = list(tello_node.keys())
        candidate_dict = {}
        tmp2 = None
        var = 1
        for key in tello_node.keys():
            target[key] = tello_node[key].get_target()
        # print('target: \n', target)
        while True:
            # for key in candidate_dict.keys():
            #     print(len(candidate_dict[key]))
            if len(candidate_list) == 0:
                # permission_list.append(candidate_list[0])
                break
            candidate_list2 = copy.deepcopy(candidate_list)
            for key in candidate_list2:
                # print(' in shide, candidate list2 is ', candidate_list2)
                for key2 in candidate_list:
                    # print(' in shide, candidate list is ', candidate_list)
                    if key == key2:
                        pass
                    else:
                        d = np.linalg.norm(np.array(target[key][0:2]) - np.array(target[key2][0:2]), 2)
                        if d <= 200:
                            tmp.append(key2)
                if len(tmp) == 0:
                    permission_list.append(key)
                    candidate_list.remove(key)
                else:
                    candidate_dict[key] = tmp
                    # print(' in shide, candidate dict is ', candidate_dict)
                tmp = []
            var = 1
            for key in candidate_list:
                if len(candidate_dict[key]) >= var:
                    tmp2 = key
                    # print('inside tmp2 is', tmp2)
                    var = len(candidate_dict[key])
            # print('outside tmp2 is', tmp2)
            if tmp2 is not None:
                ban_list.append(tmp2)
                if tmp2 in candidate_list:
                    candidate_list.remove(tmp2)
            # print('permission list', permission_list)
            # print('candidate list', candidate_list)
            # print('ban list', ban_list)
            time.sleep(0.1)
        # print('permission list', permission_list)
        # print('candidate list', candidate_list)
        # print('ban list', ban_list)
        for key in permission_list:
            permission_flag[key].value = 1
            # print('permission flag of {} is {}'.format(key, permission_flag[key].value))
        for key in ban_list:
            if key is None:
                pass
            else:
                permission_flag[key].value = 0
        # for key in permission_flag.keys():
        #     print('key: {}, value:{}'.format(key, permission_flag[key].value))
        time.sleep(0.5)


def received_ok(kwargs):
    soc_res = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    soc_res.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    soc_res.bind(('', 8889))
    while True:
        try:
            response, ip = soc_res.recvfrom(1024)
            ip = ''.join(str(ip[0]))
            res = response.decode(encoding='utf-8', errors='ignore').upper()
            if res == 'OK' or res == 'out of range':
                with kwargs[ip].get_lock():
                    kwargs[ip].value = 1
                    # print('in received ok, set res of {} from 0 to {}'.format(ip, kwargs[ip].value))
            else:
                print('RES from {}:{}'.format(ip, response.decode(encoding='utf-8', errors='ignore')))
            # print('in received ok, main alg {}'.format(main_flag.value))
        except Exception:
            print('Exception occur in received_ok thread...')


path1 = [[240, 0, 240, 0],
         [500, 0, 240, 90],
         [500, 700, 240, 180],
         [240, 700, 240, 270],
         [240, 600, 240, 270]]

num = 4
Node = {}
Res_flag = {}
Permission_flag = {}
Target = {}
scanner = Scanner()
scanner.find_available_tello(num)
tello_list = scanner.get_tello_info()
main_thread_flag = multiprocessing.Value('i', 0)
for tello in tello_list:
    Res_flag[tello[0]] = multiprocessing.Value('i', 0)
    Permission_flag[tello[0]] = multiprocessing.Value('i', 0)
rec_thread = multiprocessing.Process(target=received_ok, args=(Res_flag,), daemon=True)
rec_thread.start()
for i in range(num):
    Node[tello_list[i][0]] = TelloNode(tello_list[i], Res_flag[tello_list[i][0]],
                                       main_thread_flag, Permission_flag[tello_list[i][0]], 0)
    Node[tello_list[i][0]].init_path(path1, [240, 80, 85, 270])
    Node[tello_list[i][0]].run()
per_thread = multiprocessing.Process(target=scheduler, args=(Node, Permission_flag), daemon=True)
per_thread.start()
old = time.time()
old1 = time.time()
old2 = time.time()
face_flag = 0
try:
    while True:
        # print('in main, target:', Node[tello_list[0][0]].get_target())
        for i in range(len(tello_list)):
            if Node[tello_list[i][0]].get_thread_flag() == 1:
                del tello_list[i]
        if len(tello_list) == 0:
            print('no node alive, stop the program.')
            main_thread_flag.Value = 1
            time.sleep(1)
            break
        if face_flag == 0:
            if time.time() - old >= 5:
                for i in range(len(tello_list)):
                    if Node[tello_list[i][0]].get_path_status() == 1:
                        Node[tello_list[i][0]].update_path(path1)
                    # tmp = Node[tello_list[i][0]].get_up_camera_image()
                    # if tmp is not None:
                    #     pose = [str(i) for i in tmp['pose']]
                    #     pose = '-'.join(pose)
                    #     cv.imwrite('./'+pose+'.jpg', tmp['image'])
                old = time.time()
            # if time.time() - old2 >= 1:
            #     for i in range(len(tello_list)):
            #         f_point, pose = Node[tello_list[i][0]].get_face_point()
            #         if f_point is not None:
            #             print("get a face. f_point:{}, pose:{}".format(f_point, pose))
            #             face_flag = 1
            #             print("drone {} monitoring.".format(tello_list[i][0]))
            #             old2 = time.time()
        # else:
        #     if time.time() - old2 >= 1:
        #         for i in range(len(tello_list)):
        #             f_point, pose = Node[tello_list[i][0]].get_face_point()
        #             if f_point is not None:
        #                 print("get a face. f_point:{}, pose:{}".format(f_point, pose))
        #                 path = [pose]
        #                 path[0][0] = path[0][0] + 50
        #                 path[0][2] = path[0][2] - 75
        #                 x = path[0][0]
        #                 z = path[0][2]
        #                 Node[tello_list[i][0]].update_path(path)
        #                 c_point = [f_point[1]/2+f_point[3]/2, f_point[0]/2+f_point[2]/2]
        #                 while abs(c_point[0] - 324) > 50:
        #                     print("in y:", c_point[0] - 324)
        #                     f_point, pose = Node[tello_list[i][0]].get_face_point()
        #                     if f_point is not None:
        #                         print("get a face. f_point:{}, pose:{}".format(f_point, pose))
        #                         path = [pose]
        #                         path[0][2] = z
        #                         path[0][0] = x
        #                         if c_point[0] - 324 > 0:
        #                             path[0][1] = path[0][1] - 30
        #                         else:
        #                             path[0][1] = path[0][1] + 30
        #                         print("update path:", path)
        #                         Node[tello_list[i][0]].update_path(path)
        #                         c_point = [f_point[1] / 2 + f_point[3] / 2, f_point[0] / 2 + f_point[2] / 2]
        #                     time.sleep(3)
        #                 # x = path[0][0]
        #                 y = path[0][1]
        #                 z = path[0][2]
        #                 # d = 921.17 * 22.5 / np.linalg.norm([f_point[1] - f_point[3], f_point[0] - f_point[2]])
        #                 d = np.linalg.norm([f_point[1] - f_point[3], f_point[0] - f_point[2]])
        #                 print(d)
        #                 while d < 50:
        #                     print("in x, d:", d)
        #                     f_point, pose = Node[tello_list[i][0]].get_face_point()
        #                     if f_point is not None:
        #                         print("get a face. f_point:{}, pose:{}".format(f_point, pose))
        #                         path = [pose]
        #                         # path[0][0] = x
        #                         path[0][1] = y
        #                         path[0][2] = z
        #                         d = np.linalg.norm([f_point[1] - f_point[3], f_point[0] - f_point[2]])
        #                         print("D:", d)
        #                         if d < 50:
        #                             path[0][0] = path[0][0] + 30
        #                         else:
        #                             path[0][0] = path[0][0] - 30
        #                         print("update path:", path)
        #                         Node[tello_list[i][0]].update_path(path)
        #                         c_point = [f_point[1] / 2 + f_point[3] / 2, f_point[0] / 2 + f_point[2] / 2]
        #                     time.sleep(3)
        #                 print("drone {} at {} locking target.".format(tello_list[i][0], path[0]))
        #                 tmp = Node[tello_list[i][0]].get_up_camera_image()
        #                 if tmp is not None:
        #                     pose = [str(i) for i in path]
        #                     pose = '-'.join(pose)
        #                     cv.imwrite('./'+pose+'.jpg', tmp['image'])
        #                 f_point, pose = Node[tello_list[i][0]].get_face_point()
        #                 if f_point is not None:
        #                     d = 921.17 * 22.5 / np.linalg.norm([f_point[1]-f_point[3], f_point[0] - f_point[2]])
        #                     print("D:", d)
        #                 time.sleep(20)
        #                 for i in range(len(tello_list)):
        #                     Node[tello_list[i][0]].send_command('>streamoff')
        #                     time.sleep(0.5)
        #                     Node[tello_list[i][0]].send_command('>land')
        #                 print('landing....')
        #                 main_thread_flag.value = 1
        #                 break
        #         old2 = time.time()
        if time.time() - old1 >= 300:
            for i in range(len(tello_list)):
                Node[tello_list[i][0]].send_command('>streamoff')
                time.sleep(0.5)
                Node[tello_list[i][0]].send_command('>land')
            print('landing....')
            main_thread_flag.value = 1
            break
        time.sleep(1)
    print('main thread died!')
except KeyboardInterrupt as e:
    for i in range(len(tello_list)):
        Node[tello_list[i][0]].send_command('>streamoff')
        time.sleep(0.5)
        Node[tello_list[i][0]].send_command('>land')
    for i in range(len(tello_list)):
        time.sleep(0.5)
        Node[tello_list[i][0]].send_command('>land')




# -*- coding: utf-8 -*-
# @Time    : 2021/1/1 下午4:50
# @Author  : JakeShihao Luo
# @Email   : jakeshihaoluo@gmail.com
# @File    : main.py
# @Software: PyCharm

from scanner import *
from tello_node import *
import multiprocessing
import numpy as np
import copy
import re


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
                for key2 in candidate_list:
                    if key == key2:
                        pass
                    else:
                        d = np.linalg.norm(np.array(target[key][0:2]) - np.array(target[key2][0:2]), 2)
                        if d <= 150:
                            tmp.append(key2)
                if len(tmp) == 0:
                    permission_list.append(key)
                    candidate_list.remove(key)
                else:
                    candidate_dict[key] = tmp
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
            elif 'A' in res and 'G' in res and 'X' in res:
                t = re.split(':|;', res)
                print(t[1:-1:2])
            else:
                print('RES from {}:{}'.format(ip, response.decode(encoding='utf-8', errors='ignore')))
            # print('in received ok, main alg {}'.format(main_flag.value))
        except Exception:
            print('Exception occur in received_ok thread...')


# path1 = [[240, 0, 240, 0],
#          [500, 0, 240, 90],
#          [500, 700, 240, 180],
#          [240, 700, 240, 270]]

path2 = [[240, 0, 240, 0],
         [500, 0, 240, 90],
         [500, 350, 240, 180],
         [240, 350, 240, 270]]

path1 = [[600, 700, 240, 90],
         [600, 800, 240, 180],
         [240, 800, 240, 270],
         [240, 500, 240, 0],
         [600, 500, 240, 90]]

path3 = [[240, 0, 240, 0],
         [500, 0, 240, 90],
         [500, 350, 240, 180],
         [240, 350, 240, 270]]

path4 = [[600, 700, 240, 90],
         [600, 800, 240, 180],
         [240, 800, 240, 270],
         [240, 500, 240, 0],
         [600, 500, 240, 90]]

path = [path1, path2, path3, path4]
num = 1
Node = {}
Res_flag = {}
Permission_flag = {}
Target = {}
scanner = Scanner('192.168.1.')
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
    # ini_pose = path[i][0].copy()
    # ini_pose[2] = 85
    Node[tello_list[i][0]].init_path(path[i], [240, 100, 85, 270])
    Node[tello_list[i][0]].run()
    # Node[tello_list[i][0]].takeoff()
per_thread = multiprocessing.Process(target=scheduler, args=(Node, Permission_flag), daemon=True)
per_thread.start()
old = time.time()
old1 = time.time()
old2 = time.time()
face_flag = 0
target_pose = None
lock_flag = np.zeros(num)
# Node[tello_list[0][0]].send_command('>streamon')
while True:
    Node[tello_list[0][0]].send_command('>acceleration?')
    # Node[tello_list[0][0]].send_command('>attitude?')
    time.sleep(0.1)

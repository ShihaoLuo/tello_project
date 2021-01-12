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


def received_ok(kwargs, main_flag):
    soc_res = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    soc_res.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    soc_res.bind(('', 8889))
    while True:
        try:
            response, ip = soc_res.recvfrom(1024)
            ip = ''.join(str(ip[0]))
            if response.decode(encoding='utf-8', errors='ignore').upper() == 'OK':
                with kwargs[ip].get_lock():
                    kwargs[ip].value = 1
                    # print('in received ok, set res of {} from 0 to {}'.format(ip, kwargs[ip].value))
            else:
                print('RES from {}:{}'.format(ip, response.decode(encoding='utf-8', errors='ignore')))
            # print('in received ok, main alg {}'.format(main_flag.value))
        except Exception:
            print('Exception occur in received_ok thread...')


path1 = [[-350, 0, 100, 0],
        [-250, 0, 100, 90],
        [-250, 100, 100, 90],
        [-250, 200, 100, 90],
        [-250, 300, 100, 90],
        [-250, 400, 100, 90],
        [-250, 500, 100, 90],
        [-250, 600, 100, 180],
        [-350, 600, 100, 180],
        [-450, 600, 100, 180],
        [-550, 600, 100, 270],
        [-550, 500, 100, 270],
        [-550, 400, 100, 270],
        [-550, 300, 100, 270],
        [-550, 200, 100, 270],
        [-550, 100, 100, 270],
        [-550, 0, 100, 0],
        [-450, 0, 100, 0]]
num = 1
Node = {}
Res_flag = {}
scanner = Scanner()
scanner.find_available_tello(num)
tello_list = scanner.get_tello_info()
main_thread_flag = multiprocessing.Value('i', 0)
for tello in tello_list:
    Res_flag[tello[0]] = multiprocessing.Value('i', 0)
rec_thread = multiprocessing.Process(target=received_ok, args=(Res_flag, main_thread_flag, ), daemon=True)
rec_thread.start()
for i in range(num):
    Node[tello_list[i][0]] = TelloNode(tello_list[i], Res_flag, main_thread_flag)
    Node[tello_list[i][0]].init_path(path1, [-450, 0, 0, 0])
    Node[tello_list[i][0]].run()
old = time.time()
old1 = time.time()
try:
    while True:
        for i in range(len(tello_list)):
            if Node[tello_list[i][0]].get_thread_flag() == 1:
                del tello_list[i]
        if len(tello_list) == 0:
            print('no node alive, stop the program.')
            main_thread_flag.Value = 1
            time.sleep(1)
            break
        if time.time() - old >= 5:
            for i in range(len(tello_list)):
                Node[tello_list[i][0]].update_path(path1)
            old = time.time()
        if time.time() - old1 >= 160:
            for i in range(len(tello_list)):
                Node[tello_list[i][0]].send_command('>streamoff')
                time.sleep(0.5)
                Node[tello_list[i][0]].send_command('>land')
            print('landing....')
            main_thread_flag.value = 1
            break
except KeyboardInterrupt as e:
    for i in range(len(tello_list)):
        Node[tello_list[i][0]].send_command('>streamoff')
        time.sleep(0.5)
        Node[tello_list[i][0]].send_command('>land')




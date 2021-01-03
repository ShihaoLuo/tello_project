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


def received_ok():
    soc_res = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    soc_res.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    soc_res.bind(('', 8889))
    while True:
        try:
            if stop_flag.empty() is False:
                break
            response, ip = soc_res.recvfrom(1024)
            ip = ''.join(str(ip[0]))
            if response.decode(encoding='utf-8', errors='ignore').upper() == 'OK':
                cmd_res.put(ip)
            time.sleep(0.00)
        except socket.error as exc:
            print("[Exception_Error(rev)]Caught exception socket.error : %s\n" % exc)


path1 = [[-350, 0, 100, 0],
         [-250, 0, 100, 0],
         [-350, 0, 100, 0],
         [-450, 0, 100, 0],
         [-550, 0, 100, 0],
         [-450, 0, 100, 0]]
         # [-350, 0, 100, 0],
         # [-250, 0, 100, 0],
         # [-350, 0, 100, 0],
         # [-450, 0, 100, 0],
         # [-550, 0, 100, 0],
         # [-450, 0, 100, 0],
         # [-350, 0, 100, 0],
         # [-250, 0, 100, 0],
         # [-350, 0, 100, 0],
         # [-450, 0, 100, 0]]
num = 1
Node = {}
cmd_res = multiprocessing.Queue()
stop_flag = multiprocessing.Queue()
scanner = Scanner()
scanner.find_available_tello(num)
tello_list = scanner.get_tello_info()
rec_thread = multiprocessing.Process(target=received_ok)
rec_thread.start()
for i in range(num):
    Node[tello_list[i][0]] = TelloNode(tello_list[i])
    Node[tello_list[i][0]].init_path(path1, [-450, 0, 0, 0])
    Node[tello_list[i][0]].run()
try:
    while True:
        if cmd_res.empty() is False:
            tmp = cmd_res.get()
            print('get ok')
            Node[tmp].update_res()
            time.sleep(0.01)
        for i in range(len(tello_list)):
            if Node[tello_list[i][0]].get_thread_flag() == 1:
                del tello_list[i]
        if len(tello_list) == 0:
            print('no node alive, stop the program.')
            stop_flag.put(1)
            time.sleep(1)
            break
except KeyboardInterrupt as e:
    pass



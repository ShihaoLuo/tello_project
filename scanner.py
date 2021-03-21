# -*- coding: utf-8 -*-
# @Time    : 2021/1/1 下午4:22
# @Author  : JakeShihao Luo
# @Email   : jakeshihaoluo@gmail.com
# @File    : scanner.py
# @Software: PyCharm

import socket
import time
import threading
import inspect
import ctypes


class Scanner:
    def __init__(self, subnet):
        self.host_ip = None
        self.local_ip = ''
        self.local_port = 8889
        self.video_port_base = 20000
        self.ctr_port_base = 25000
        self.tello_ip_list = []
        self.tello_list = []
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind((self.local_ip, self.local_port))
        self.receive_thread = threading.Thread(target=self._receive_thread)
        self.receive_thread.setDaemon(True)
        self.receive_thread.start()
        self.tello_info = []
        self.subnet = subnet

    def get_host_ip(self):
        _s = None
        try:
            _s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            _s.connect(('8.8.8.8', 80))
            self.host_ip = _s.getsockname()[0]
        finally:
            _s.close()

    def find_available_tello(self, num):
        print('[Start_Searching]Searching for %s available Tello...\n' % num)
        possible_addr = []
        for i in range(100, 250, 1):
            possible_addr.append(self.subnet+str(i))
        while len(self.tello_ip_list) < num:
            print('[Still_Searching]Trying to find Tello in subnets...\n')
            for tello_ip in self.tello_ip_list:
                if tello_ip in possible_addr:
                    possible_addr.remove(tello_ip)
            for ip in possible_addr:
                if ip == self.host_ip:
                    continue
                self.socket.sendto('command'.encode('utf-8'), (ip, 8889))
            time.sleep(0.1)
        self.stop_thread(self.receive_thread)
        self.socket.close()

    def _receive_thread(self):
        while True:
            try:
                self.response, ip = self.socket.recvfrom(1024)
                ip = ''.join(str(ip[0]))
                # print(self.response.decode()=='ok')
                if self.response.decode(encoding='utf-8',
                                        errors='ignore').upper() == 'OK' and ip not in self.tello_ip_list:
                    v_port = self.video_port_base + int(str(ip).split('.')[3])
                    c_port = self.ctr_port_base + int(str(ip).split('.')[3])
                    self.socket.sendto(('port 8890 ' + str(v_port)).encode('utf-8'), (ip, 8889))
                    self.response, _ = self.socket.recvfrom(1024)
                    if self.response.decode(encoding='utf-8', errors='ignore').upper() == 'OK':
                        self.socket.sendto('battery?'.encode('utf-8'), (ip, 8889))
                        self.response, _ = self.socket.recvfrom(1024)
                        res, _ = self.socket.recvfrom(1024)
                        try:
                            battery_value = int(res.decode())
                            if battery_value >= 20:
                                self.tello_ip_list.append(ip)
                                self.tello_info.append((ip, c_port, v_port))
                                print('[Found_Tello]Found Tello.The Tello ip is:%s, Control port is:%d, Video port is:%d'
                                      % (ip, c_port, v_port))
                                print(' Battery: ', battery_value)
                            else:
                                print('Battery low: ', battery_value)
                        except Exception as e:
                            pass
            except socket.error as exc:
                print("[Exception_Error(rev)]Caught exception socket.error : %s\n" % exc)
                break

    def get_tello_info(self):
        return self.tello_info

    def _async_raise(self, tid, exctype):
        """raises the exception, performs cleanup if needed"""
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

    def stop_thread(self, thread):
        self._async_raise(thread.ident, SystemExit)
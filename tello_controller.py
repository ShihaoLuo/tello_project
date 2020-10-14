#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 10:28:41 2020

@author: jakeluo
"""

# -*- coding: utf-8 -*-
# import sys
import time
import tello_manager
import queue
# import time
import os
# import binascii
# import threading
from threading import Thread


class Tell_Controller:
    def __init__(self):
        self.manager = tello_manager.Tello_Manager()
        self.manager.thread_start()
        self.start_time = str(time.strftime("%a-%d-%b-%Y_%H-%M-%S-%Z",
                                            time.localtime(time.time())))
        self.tello_list = []
        self.execution_pools = []
        self.sn_ip_dict = {}
        self.id_sn_dict = {}
        self.ip_fid_dict = {}
        self.sn_list = []

    def create_execution_pools(self, num):
        pools = []
        for x in range(num):
            execution_pool = queue.Queue()
            pools.append(execution_pool)
        return pools

    def drone_handler(self, tello, _queue):
        while True:
            while _queue.empty():
                pass
            _command = _queue.get()
            # print("Send command: %s"%command)
            tello.send_command(_command)

    def all_queue_empty(self, _execution_pools):
        for Queue in _execution_pools:
            if not Queue.empty():
                return False
        return True

    def all_got_response(self, _manager):
        for tello_log in _manager.get_log().values():
            if not tello_log[-1].got_response():
                return False
        return True

    def save_log(self, _manager):
        log = _manager.get_log()
        if not os.path.exists('log'):
            try:
                os.makedirs('log')
            except Exception:
                pass
        out = open('log/' + self.start_time + '.txt', 'w')
        cnt = 1
        for stat_list in log.values():
            out.write('------\nDrone: %s\n' % cnt)
            cnt += 1
            for stat in stat_list:
                # stat.print_stats()
                _str = stat.return_stats()
                out.write(_str)
            out.write('\n')

    def check_timeout(self, _start_time,  _end_time, timeout):
        diff = _end_time - _start_time
        time.sleep(0.1)
        return diff > timeout

    def command(self, _command):
        if _command != '' and _command != '\n':
            _command = _command.rstrip()
            if '//' in _command:  # ignore comments
                pass
            # elif 'scan' in command:
            elif '>' in _command:
                id_list = []
                _id = _command.partition('>')[0]
                if _id == '*':
                    for x in range(len(self.tello_list)):
                        id_list.append(x)
                else:
                    # index starbattery_checkt from 1
                    id_list.append(int(_id) - 1)
                action = str(_command.partition('>')[2])
                for tello_id in id_list:
                    tmp_sn = self.id_sn_dict[tello_id]
                    reflec_ip = self.sn_ip_dict[tmp_sn]
                    fid = self.ip_fid_dict[reflec_ip]
                    self.execution_pools[fid].put(action)
            elif 'battery_check' in _command:
                threshold = int(_command.partition('battery_check')[2])
                for Queue in self.execution_pools:
                    Queue.put('battery?')
                # wait till all commands are executed
                while not self.all_queue_empty(self.execution_pools):
                    time.sleep(0.5)
                # wait for new log object append
                time.sleep(1)
                # wait till all responses are received
                while not self.all_got_response(self.manager):
                    time.sleep(0.5)
                for tello_log in self.manager.get_log().values():
                    # print(tello_log[-1].response)
                    battery = int(tello_log[-1].response)
                    print ('[Battery_Show]show drone battery: %d  ip:%s\n'
                           % (battery, tello_log[-1].drone_ip))
                    if battery < threshold:
                        print(
                            '[Battery_Low]IP:%s Battery<Threshold. Exiting.\n'
                            % tello_log[-1].drone_ip)
                        self.save_log(self.manager)
                        exit(0)
                print ('[Battery_Enough]Pass battery check\n')
            elif 'delay' in _command:
                delay_time = float(_command.partition('delay')[2])
                print (
                    '[Delay_Seconds]Start Delay for %f second\n' % delay_time)
                time.sleep(delay_time)
            elif 'correct_ip' in _command:
                for Queue in self.execution_pools:
                    Queue.put('sn?')
                while not self.all_queue_empty(self.execution_pools):
                    time.sleep(0.5)
                time.sleep(1)
                while not self.all_got_response(self.manager):
                    time.sleep(0.5)
                for tello_log in self.manager.get_log().values():
                    sn = str(tello_log[-1].response.decode())
                    self.sn_list.append(sn)
                    # print(sn)
                    tello_ip = str(tello_log[-1].drone_ip)
                    # print(tello_ip)
                    self.sn_ip_dict[sn] = tello_ip
            elif '=' in _command:
                drone_id = int(_command.partition('=')[0])
                drone_sn = _command.partition('=')[2]
                self.id_sn_dict[drone_id - 1] = drone_sn
                print ('[IP_SN_FID]:Tello_IP:%s------Tello_SN:'
                       '%s------Tello_fid:%d\n'
                       % (self.sn_ip_dict[drone_sn], drone_sn, drone_id))
                # print id_sn_dict[drone_id]
            elif 'sync' in _command:
                timeout = float(_command.partition('sync')[2])
                print ('[Sync_And_Waiting]Sync for %s seconds \n' % timeout)
                time.sleep(1)
                try:
                    start = time.time()
                    # wait till all commands are executed
                    while not self.all_queue_empty(self.execution_pools):
                        now = time.time()
                        if self.check_timeout(start, now, timeout):
                            raise RuntimeError
                    print ('[All_Commands_Send]All queue empty '
                           'and all command send,continue\n')
                    # wait till all responses are received
                    while not self.all_got_response(self.manager):
                        now = time.time()
                        if self.check_timeout(start, now, timeout):
                            raise RuntimeError
                    print ('[All_Responses_Get]All response got, continue\n')
                except RuntimeError:
                    print ('[Quit_Sync]Fail Sync:'
                           'Timeout exceeded, continue...\n')
            elif 'wait' in _command:
                wait_time = float(_command.partition('wait')[2])
                start = time.time()
                while(time.time() - start < wait_time):
                    for Queue in self.execution_pools:
                        Queue.put('command')
                        time.sleep(1)
        # wait till all commands are executed
        while not self.all_queue_empty(self.execution_pools):
            time.sleep(0.5)
        time.sleep(1)
        # wait till all responses are received
        while not self.all_got_response(self.manager):
            time.sleep(0.5)
        self.save_log(self.manager)

    def scan(self, num_of_tello):
        # num_of_tello = int(command.partition('scan')[2])
        self.manager.find_avaliable_tello(num_of_tello)
        self.tello_list = self.manager.get_tello_list()
        self.execution_pools = self.create_execution_pools(num_of_tello)
        for x in range(len(self.tello_list)):
            t1 = Thread(target=self.drone_handler,
                        args=(self.tello_list[x], self.execution_pools[x]))
            self.ip_fid_dict[self.tello_list[x].tello_ip] = x
            # str_cmd_index_dict_init_flag [x] = None
            t1.daemon = True
            t1.start()

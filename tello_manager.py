import threading
import socket
import time
import netifaces
import netaddr
from collections import defaultdict
from stats import Stats
import inspect
import ctypes


class Tello:
    """
    A wrapper class to interact with Tello
    Communication with Tello is handled by Tello_Manager
    """

    def __init__(self, tello_ip, video_port, _Tello_Manager):
        self.tello_ip = tello_ip
        self.Tello_Manager = _Tello_Manager
        self.video_port = video_port

    def send_command(self, command):
        return self.Tello_Manager.send_command(command, self.tello_ip)


class Tello_Manager:
    def __init__(self):
        self.local_ip = ''
        self.local_port = 8889
        self.state_port = 8890
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket_state = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket_state.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind((self.local_ip, self.local_port))
        self.socket_state.bind((self.local_ip, self.state_port))
        # thread for receiving cmd ack
        self.receive_thread = threading.Thread(target=self._receive_thread)
        self.receive_thread.setDaemon(True)
        self.tello_ip_list = []
        self.tello_list = []
        self.log = defaultdict(list)
        self.COMMAND_TIME_OUT = 3.0
        self.last_response_index = {}
        self.str_cmd_index = {}
        self.response = None
        self.videoportbase = 20000
        self.state_field_converters = {
                # Tello EDU with mission pads enabled only
                'mid': int,
                'x': int,
                'y': int,
                'z': int,
                # 'mpry': (custom format 'x,y,z')

                # common entries
                'pitch': int,
                'roll': int,
                'yaw': int,
                'vgx': int,
                'vgy': int,
                'vgz': int,
                'templ': int,
                'temph': int,
                'tof': int,
                'h': int,
                'bat': int,
                'baro': float,
                'time': int,
                'agx': float,
                'agy': float,
                'agz': float,
                }

    def get_host_ip(self):
        _s = None
        try:
            _s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            _s.connect(('8.8.8.8', 80))
            ip = _s.getsockname()[0]
        finally:
            _s.close()
        return ip

    def find_avaliable_tello(self, num):
        """
        Find avaliable tello in server's subnets
        :param num: Number of Tello this method is expected to find
        :return: None
        """
        print('[Start_Searching]Searching for %s available Tello...\n' % num)
        local_ip = self.get_host_ip()
        #subnets, address = self.get_subnets()
        possible_addr = []

        #for subnet, netmask in subnets:
         #   for ip in IPNetwork('%s/%s' % (subnet, netmask)):
          #      # skip local and broadcast
           #     if str(ip).split('.')[3] == '0' or str(ip).split('.')[3] == '255':
            #        continue
             #   possible_addr.append(str(ip))
        for i in range(100, 200, 1):
            possible_addr.append('192.168.50.'+str(i))
        while len(self.tello_ip_list) < num:
            print('[Still_Searching]Trying to find Tello in subnets...\n')

            # delete already fond Tello
            for tello_ip in self.tello_ip_list:
                if tello_ip in possible_addr:
                    possible_addr.remove(tello_ip)
            # skip server itself
            for ip in possible_addr:
                #if ip in address:
                 #   continue
                # record this command
                if ip == local_ip:
                    continue
                self.log[ip].append(Stats('command', len(self.log[ip])))
                self.socket.sendto('command'.encode('utf-8'), (ip, 8889))
                #print('send to {}'.format(ip))
            time.sleep(2)

        # filter out non-tello addresses in log
        temp = defaultdict(list)
        for ip in self.tello_ip_list:
            temp[ip] = self.log[ip]
        self.log = temp

    def get_subnets(self):
        """
        Look through the server's internet connection and
        returns subnet addresses and server ip
        :return: list[str]: subnets
                 list[str]: addr_list
        """
        subnets = []
        ifaces = netifaces.interfaces()
        addr_list = []
        for myiface in ifaces:
            addrs = netifaces.ifaddresses(myiface)

            if socket.AF_INET not in addrs:
                continue
            # Get ipv4 stuff
            ipinfo = addrs[socket.AF_INET][0]
            address = ipinfo['addr']
            netmask = ipinfo['netmask']

            # limit range of search. This will work for router subnets
            if netmask != '255.255.255.0':
                continue

            # Create ip object and get
            cidr = netaddr.IPNetwork('%s/%s' % (address, netmask))
            network = cidr.network
            subnets.append((network, netmask))
            addr_list.append(address)
        return subnets, addr_list

    def get_tello_list(self):
        return self.tello_list

    def send_command(self, command, ip):
        """
        Send a command to the ip address. Will be blocked until
        the last command receives an 'OK'.
        If the command fails (either b/c time out or error),
        will try to resend the command
        :param command: (str) the command to send
        :param ip: (str) the ip of Tello
        :return: The latest command response
        """
        # global cmd
        command_sof_1 = ord(command[0])
        command_sof_2 = ord(command[1])
        if command_sof_1 == 0x52 and command_sof_2 == 0x65:
            multi_cmd_send_flag = True
        else:
            multi_cmd_send_flag = False

        if multi_cmd_send_flag is True:
            self.str_cmd_index[ip] = self.str_cmd_index[ip] + 1
            for num in range(1, 5):
                str_cmd_index_h = self.str_cmd_index[ip] / 128 + 1
                str_cmd_index_l = self.str_cmd_index[ip] % 128
                if str_cmd_index_l == 0:
                    str_cmd_index_l = str_cmd_index_l + 2
                cmd_sof = [0x52, 0x65, str_cmd_index_h, str_cmd_index_l, 0x01, num + 1, 0x20]
                cmd_sof_str = str(bytearray(cmd_sof))
                cmd = cmd_sof_str + command[3:]
                self.socket.sendto(cmd.encode('utf-8'), (ip, 8889))

            print ('[Multi_Command]----Multi_Send----IP:%s----Command:   %s\n' % (ip, command[3:]))
            real_command = command[3:]
        else:
            self.socket.sendto(command.encode('utf-8'), (ip, 8889))
            print ('[Single_Command]----Single_Send----IP:%s----Command:   %s\n' % (ip, command))
            real_command = command
        start = time.time()
        print(self.log[ip][-1].got_response())
        while not self.log[ip][-1].got_response():
            now = time.time()
            diff = now - start
            if diff > self.COMMAND_TIME_OUT:
                print ('[Not_Get_Response]Max timeout exceeded...command: %s \n' % real_command)
                start = time.time()
                print('cmd in manager:', command)
                self.socket.sendto(command.encode('utf-8'), (ip, 8889))
                print('[Single_Command]----Single_Send_again----IP:%s----Command:   %s\n' % (ip, command))
        self.log[ip].append(Stats(real_command, len(self.log[ip])))

    def parse_state(self, state):
        """Parse a state line to a dictionary
        Internal method, you normally wouldn't call this yourself.
        """
        state = state.strip()
        if state == 'ok':
            return {}
        state_dict = {}
        for field in state.split(';'):
            split = field.split(':')
            if len(split) < 2:
                continue
            key = split[0]
            value = split[1]
            if key in self.state_field_converters:
                value = self.state_field_converters[key](value)
            state_dict[key] = value
        return state_dict


    def get_state(self, ip):
        try:
            data, address = self.socket_state.recvfrom(166)
            address = address[0]
            if address == ip:
                data = data.decode('ASCII')
                state = self.parse_state(data)
                return state
            else:
                return None
        except Exception as e:
            print(e)

    def _receive_thread(self):
        """Listen to responses from the Tello.

        Runs as a thread, sets self.response to whatever the Tello last returned.

        """
        while True:
            # print("the receive thread is running")
            try:
                self.response, ip = self.socket.recvfrom(1024)
                ip = ''.join(str(ip[0]))
                # print(self.response.decode()=='ok')
                if self.response.decode(encoding='utf-8',
                                        errors='ignore').upper() == 'OK' and ip not in self.tello_ip_list:
                    port = self.videoportbase + int(str(ip).split('.')[3])
                    self.socket.sendto(('port 8890 ' + str(port)).encode('utf-8'), (ip, 8889))
                    time.sleep(0.5)
                    self.response, _ = self.socket.recvfrom(1024)
                    if self.response.decode(encoding='utf-8',
                                            errors='ignore').upper() == 'OK':
                        print('[Found_Tello]Found Tello.The Tello ip is:%s.Video port is:%d\n' % (ip, port))
                        self.tello_ip_list.append(ip)
                        self.last_response_index[ip] = 100
                        self.tello_list.append(Tello(ip, port, self))
                        self.str_cmd_index[ip] = 1

                response_sof_part1 = self.response[0]
                response_sof_part2 = self.response[1]
                if response_sof_part1 == 0x52 and response_sof_part2 == 0x65:
                    response_index = self.response[3]

                    if response_index != self.last_response_index[ip]:
                        # print '--------------------------response_index:%x %x'%(response_index,self.last_response_index)
                        print('[Multi_Response] ----Multi_Receive----IP:%s----Response:   %s ----\n' % (
                            ip, self.response[7:].decode('utf-8')))
                        self.log[ip][-1].add_response(self.response[7:], ip)

                    self.last_response_index[ip] = response_index
                else:
                    print('[Single_Response]----Single_Receive----IP:%s----Response:   %s ----\n' % (ip, self.response.decode('utf-8')))
                    self.log[ip][-1].add_response(self.response, ip)
                #print('[Response_WithIP]----Receive----IP:%s----Response:%s----\n' % (ip, self.response))

            except socket.error as exc:
                print ("[Exception_Error(rev)]Caught exception socket.error : %s\n" % exc)

    def get_log(self):
        return self.log

    def close(self):
        self.socket.close()
        self.stop_thread()

    def thread_start(self):
        self.receive_thread.start()
        #self.state_receive_thread.start()

    def _async_raise(self, tid, exctype):
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
        self._async_raise(self.receive_thread.ident, SystemExit)
        #self._async_raise(self.state_receive_thread.ident, SystemExit)

import socket

my_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # socket for sending cmd
my_socket.bind(('', 8889))
cmd_str = 'setresolution low'
print ('sending command %s' % cmd_str)
my_socket.sendto(cmd_str.encode('utf-8'), ('192.168.50.99', 8889))
response, ip = my_socket.recvfrom(100)
print('from %s: %s' % (ip, response))

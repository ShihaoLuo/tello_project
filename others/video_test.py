import numpy as np
import socket
import h264decoder
from scanner import Scanner
import queue

def _h264_decode(packet_data, decoder, queue):
    frames = decoder.decode(packet_data)
    for frame_data in frames:
        (frame, w, h, ls) = frame_data
        if frame is not None:
            frame = np.fromstring(frame, dtype=np.ubyte, count=len(frame), sep='')
            frame = (frame.reshape((h, ls // 3, 3)))
            frame = frame[:, :w, :]
            while queue.qsize() > 1:
                queue.get()
            queue.put(frame)

def _receive_video_thread(video_socket, decoder, queue):
    pack_data = ''
    print("receive video thread start....")
    while True:
        try:
            res_string, ip = video_socket.recvfrom(2048)
            pack_data += res_string.hex()
            if len(res_string) != 1460:
                tmp = bytes.fromhex(pack_data)
                _h264_decode(tmp, decoder , queue)
                pack_data = ''
        except socket.error as exc:
            print("Caught exception socket.error(video_thread): %s" % exc)

scanner = Scanner('192.168.1.')
scanner.find_available_tello(1)
tello_list = scanner.get_tello_info()
print(tello_list)
video_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
video_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 10 * 1024)
video_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
video_socket.bind(('', tello_list[0][2]))
soc = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
soc.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
soc.bind(('', tello_list[0][1]))
soc.sendto('streamon'.encode('utf-8'), (tello_list[0][0], 8889))
queue = queue.Queue()
decoder = h264decoder.H264Decoder()
_receive_video_thread(video_socket, decoder, queue)
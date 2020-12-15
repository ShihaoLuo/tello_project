import numpy as np
import time
from pose_estimater import  pose_estimater
import threading

class Scheduler:
    def __init__(self, controller, video):
        self.controller = controller
        self.video = video
        self.pose_estimater = pose_estimater.PoseEstimater('SIFT', 15)
        self.pose_estimater.loaddata('pose_estimater/dataset/')
        self.tello_list = controller.tello_list
        self.path = {}
        for tello in self.tello_list:
            self.path[tello.tello_ip] = []
        self.id_sn = controller.id_sn_dict
        self.sn_ip = controller.sn_ip_dict
        self.id_ip = {}
        for key in self.id_sn.keys():
            self.id_ip[key] = self.sn_ip[self.id_sn[key]]
        self.pose = {}
        self.permission = {}
        for key in self.id_ip.keys():
            self.permission[key] = 0


    def drone_init(self):
        for i in self.id_ip.keys():
            self.controller.command(str(i)+'>setfps high')
            self.controller.command(str(i)+'>setresolution high')
            self.controller.command(str(i)+'>setbitrate 5')
            self.controller.command(str(i)+'>streamon')
#
    def init_path(self, loopcount, id, path):
        tmp = np.array(path)
        self.path[self.id_ip[id]] = np.tile(tmp, (loopcount, 1))
        self.pose[self.id_ip[id]] = path[0]
        self.pose[self.id_ip[id]][2] = 0

    def updatepos(self, _last_cmd, _last_pose, _Video):
        img = _Video.get_frame()
        pose = np.array([0, 0, 0, 0])
        if img is not None:
            _pose, yaw = self.pose_estimater.estimate_pose(img)
            if _pose is not None:
                print('in img1')
                time.sleep(2.5)
                while True:
                    img = _Video.get_frame()
                    if img is not None:
                        break
                _pose, yaw = self.pose_estimater.estimate_pose(img)
                if _pose is not None:
                    print('in img2')
                    pose[0] = _pose[0]
                    pose[1] = _pose[1]
                    pose[2] = _pose[2]
                    pose[3] = yaw
                    return pose
        if 'ccw' in _last_cmd:
            angle = float(_last_cmd.partition(' ')[2])
            print('angle:{}'.format(angle))
            pose[3] = angle + _last_pose[3]
            pose[0:3] = _last_pose[0:3]
        elif 'go' in _last_cmd:
            tmp = _last_cmd.split(' ')[1:4]
            tmp = [int(i) for i in tmp]
            alpha = _last_pose[3] * 3.1416 / 180
            M = np.array([[np.cos(alpha), np.sin(alpha), 0],
                          [-np.sin(alpha), np.cos(alpha), 0],
                          [0, 0, 1]])
            tmp = np.dot(np.linalg.inv(M), tmp)
            tmp = np.append(tmp, 0)
            pose = _last_pose + tmp
        else:
            pose = _last_pose
        return pose

    def start(self):
        run_thread = {}
        for _id in self.id_ip.keys():
            run_thread[_id] = threading.Thread(target=self.run, args=(_id,))
            run_thread[_id].setDaemon(True)
            run_thread[_id].start()
        if len(self.id_ip) == 1:
            self.permission[self.id_ip.keys()[0]] = 1
        else:
            for _id in self.id_ip.keys():
                for _id2 in self.id_ip.keys():
                    if _id == _id2:
                        pass
                    else:
                        d = np.linalg.norm(self.pose[self.id_ip[_id]][0:3] - self.pose[self.id_ip[_id2][0:3]], 2)
                        if d < 100:
                            self.permission[_id] = 1


    def run(self, _id):
        if self.permission[_id] == 1:
            if self.pose[self.id_ip[_id]][2] == 0:
                self.controller.command(str(_id) + '>takeoff')
                self.controller.command(str(_id) + '>up 170')
                self.pose[self.id_ip[_id]][2] = 70
                time.sleep(2.5)
                self.pose[self.id_ip[_id]] = self.updatepos(' ', self.pose[self.id_ip[_id]], self.video)
            for target in self.path[self.id_ip[_id]]:
                print("--------------------------")
                print("target:{}".format(target))
                theta = target[3] - self.pose[self.id_ip[_id]][3]
                if abs(theta) > 30:
                    cmd = 'ccw ' + str(theta)
                    self.controller.command(str(_id)+">"+cmd)
                    self.pose[self.id_ip[_id]] = self.updatepos(cmd, self.pose[self.id_ip[_id]], self.video)
                    print("Pose in the drone_world is {}".format(self.pose[self.id_ip[_id]]))
                if np.linalg.norm(target[0:3] - self.pose[self.id_ip[_id]][0:3]) < 50:
                    pass
                else:
                    alpha = self.pose[self.id_ip[_id]][3] * 3.1416 / 180
                    #print("alpha:{}".format(alpha))
                    M = np.array([[np.cos(alpha), np.sin(alpha), 0],
                                  [-np.sin(alpha), np.cos(alpha), 0],
                                  [0, 0, 1]])
                    #print("M: {}".format(M))
                    tmp = target[0:3]-self.pose[self.id_ip[_id]][0:3]
                    tmp = np.dot(M, tmp)
                    #tmp = [int(i) for i in tmp]
                    tmp = np.append(tmp, 100)
                    tmp = [int(i) for i in tmp]
                    tmp = [str(i) for i in tmp]
                    cmd = 'go ' + ' '.join(tmp)
                    self.controller.command(str(_id) + '>' + cmd)
                    self.pose[self.id_ip[_id]] = self.updatepos(cmd, self.pose[self.id_ip[_id]], self.video)
                    print("Pose in the drone_world is {}".format(self.pose[self.id_ip[_id]]))
            else:
                self.controller.command(str(_id) + '>wait 5')

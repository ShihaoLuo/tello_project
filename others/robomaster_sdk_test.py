import robomaster
from robomaster import robot

tl = robot.Drone()
robomaster.config.LOCAL_IP_STR = '192.168.1.101'
tl.initialize('sta')
robomaster.flight
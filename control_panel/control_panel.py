import cv2 as cv
import numpy as np
import threading
import time
import ctypes
import inspect

class CtrPanel:
    def __init__(self, controller):
        self.controller = controller
        self.ui = np.zeros((500, 500))
        self.x = 0
        self.y = 0
        self.z = 0
        self.yaw = 0
        cv.putText(self.ui, 'W', (200, 100), cv.FONT_HERSHEY_PLAIN, 2.0, (255, 255, 255))
        cv.putText(self.ui, 'S', (200, 200), cv.FONT_HERSHEY_PLAIN, 2.0, (255, 255, 255))
        cv.putText(self.ui, 'A', (100, 200), cv.FONT_HERSHEY_PLAIN, 2.0, (255, 255, 255))
        cv.putText(self.ui, 'D', (300, 200), cv.FONT_HERSHEY_PLAIN, 2.0, (255, 255, 255))
        cv.putText(self.ui, 'Q', (120, 110), cv.FONT_HERSHEY_PLAIN, 2.0, (255, 255, 255))
        cv.putText(self.ui, 'E', (280, 110), cv.FONT_HERSHEY_PLAIN, 2.0, (255, 255, 255))
        cv.putText(self.ui, 'esc:quit', (300, 300), cv.FONT_HERSHEY_PLAIN, 2.0, (255, 255, 255))
        cv.putText(self.ui, 'l:land', (300, 350), cv.FONT_HERSHEY_PLAIN, 2.0, (255, 255, 255))
        self.diaplsy_thread = threading.Thread(target=self.displayui, args=())
        self.state_thread = threading.Thread(target=self.compu_state, args=())

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
        self._async_raise(self.state_thread.ident, SystemExit)
        #self._async_raise(self.state_receive_thread.ident, SystemExit)

    def displayui(self):
        #self.state_thread.start()
        now = time.time()
        while True:
            img = self.ui.copy()
            img = self.updateui(img)
            cv.imshow('control panel', img)
            key = cv.waitKey(50) & 0xff
            if key == 27:
                cv.destroyWindow('control panel')
                break
            elif key == ord('w'):
                self.controller.command('*>forward 50')
            elif key == ord('s'):
                self.controller.command('*>back 50')
            elif key == ord('a'):
                self.controller.command('*>left 50')
            elif key == ord('d'):
                self.controller.command('*>right 50')
            elif key == ord('q'):
                self.controller.command('*>ccw 15')
            elif key == ord('e'):
                self.controller.command('*>cw 15')
            elif key == ord('l'):
                self.controller.command('*>land')
            else:
                if time.time() - now >= 4.5:
                    self.controller.command('*>command')
                    now = time.time()
        self.stop_thread()

    def start(self):
        self.diaplsy_thread.start()
        self.diaplsy_thread.join()

    def join(self):
        self.diaplsy_thread.join()


    def compu_state(self):
        while True:
            try:
                now = time.time()
                state = self.controller.get_state(1)
                self.x += state['vgx']*0.2
                self.y += state['vgy']*0.2
                self.yaw = state['yaw']
                print('x: {}'.format(self.x))
                time.sleep(0.2 - time.time() + now)
            except Exception as e:
                print(e)

    def updateui(self, img):
        #state = self.controller.get_state(1)
        cv.putText(img, 'vx:' + str(self.x), (100, 350), cv.FONT_HERSHEY_PLAIN, 2.0, (255, 255, 255))
        cv.putText(img, 'vy:' + str(self.y), (100, 400), cv.FONT_HERSHEY_PLAIN, 2.0, (255, 255, 255))
        cv.putText(img, 'yaw:' + str(self.yaw), (100, 450), cv.FONT_HERSHEY_PLAIN, 2.0, (255, 255, 255))
        return img

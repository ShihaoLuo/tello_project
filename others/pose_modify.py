# -*- coding: utf-8 -*-
# @Time    : 2021/2/2 下午3:28
# @Author  : JakeShihao Luo
# @Email   : jakeshihaoluo@gmail.com
# @File    : pose_m0dify.py.py
# @Software: PyCharm

from pose_estimater.pose_estimater import *

po = PoseEstimater(min_match=15)
po.loaddata('pose_estimater/dataset/')
po.showdataset()
# po.modifydata('wall', True, False)
# coding: utf-8

'''
Created on 26 nov 2025
Modified on 28 oct 2025

@author: Fabien Bonardi
'''

import random
from copy import deepcopy
import numpy as np
from scipy.ndimage import rotate


class RobotSim:
    '''
    classdocs
    '''
    def __init__(self, **params):
        self.x = params.get('x', 30)
        self.y = params.get('y', 30)
        self.theta = params.get('theta', 0)
        self.sigmaDTheta = params.get('sigmaDTheta', 1)
        self.sigmaDx = params.get('sigmaDx', 0.5)
        self.sigmaSensor = params.get('sigmaSensor', 1)
        self.map = self.generateMap()

    def commandAndGetData(self, dx, dtheta):
        # theta in degrees
        self.theta += dtheta + np.random.normal(scale=self.sigmaDTheta)
        if self.theta < -180:
            self.theta += 360
        if self.theta > 180:
            self.theta -= 360
        dxTrue = dx + np.random.normal(scale=self.sigmaDx)
        self.x += np.sin(np.radians(self.theta))*dxTrue
        self.y += np.cos(np.radians(self.theta))*dxTrue
        # print(f"val x {self.x} val y {self.y} val theta {self.theta}")
        # if self.map[int(self.x), int(self.y)] == 1:
        #     raise Exception("CRASH ON OBSTACLE!")
        if int(self.x) < 0 or int(self.x) >= 500 or int(self.y) < 0 or int(self.y) >= 500:
            raise Exception("CRASH OUT OF BOUNDS!")
        if int(self.x) == 450 and int(self.y) == 450:
            raise Exception("Goal reached!")
        return self.generateData(), (self.x, self.y, self.theta)

    def generateMap(self, map_size=500):
        map = np.zeros((map_size, map_size))
        wBound = 25
        map[:wBound,:] = 1
        map[:,-wBound:] = 1
        map[-wBound:,:] = 1
        map[:,:wBound] = 1
        for i in range(50):
            xObs = int(random.random()*(map_size-100)+50)
            yObs = int(random.random()*(map_size-100)+50)
            map[xObs-10:xObs+11, yObs-10:yObs+11] = 1
        return map

    def generateData(self):
        xt = int(self.x)
        yt = int(self.y)
        sensorImage = self.map[xt-25:xt+25, yt-25:yt+25]
        sensorImage = rotate(sensorImage,
                             angle=self.theta
                                    + 90
                                    + np.random.normal(scale=self.sigmaSensor),
                             reshape=False)
        sensorImage = 0.99*sensorImage + 0.01*np.random.random((50, 50))
        return sensorImage
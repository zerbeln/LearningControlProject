#!/usr/bin/env python

import numpy as np
import math
from math import sin, cos


class Agent:

    def __init__(self, p):
        self.sensor_res = p.sensor_resolution
        self.lidar_sensors = np.zeros(self.sensor_res)
        self.agent_pos = np.zeros(3)  # x, y, theta
        self.agent_start_pos = np.zeros(3)  # x, y, theta
        self.n_inputs = p.num_inputs
        self.body_radius = p.agent_rad
        self.sensor_radius = p.detection_radius
        self.max_vel = p.max_vel
        self.max_rot_vel = p.max_rot_vel
        self.buffer = p.min_dist_to_wall
        self.collision_penalty = p.coll_penalty
        self.step_penalty = p.stp_penalty
        self.set_agent_start_pos()

    def set_agent_start_pos(self):
        """
        Gives the agent a new starting position in the world (Complete world reset)
        :return:
        """
        self.agent_start_pos = [10.0,10.0,0.0]
        self.agent_pos = self.agent_start_pos

    def reset_agent_to_start(self):
        """
        Resets agent to its initial position in the world
        :return:
        """
        self.agent_pos = self.agent_start_pos

    def lidar_scan(self, world_x, world_y, walls):
        """
        This function is called when the agent needs to gather LIDAR inputs for the NN
        :return: vector of lidar scan with input size
        """
        # reset lidar each scan
        self.lidar_sensors = np.zeros(self.sensor_res)
        wallDict = {}  # init wall/boundary dictionary
        # build boundary segments
        # format [X3,X4,Y3,Y4]
        wallDict["worldLeft"] = [0,0,0,world_y]
        wallDict["worldRight"] = [world_x,world_x,0,world_y]
        wallDict["worldTop"] = [0,world_x,world_y,world_y]
        wallDict["worldBottom"] = [0,world_x,0,0]

        x1 = self.agent_pos[0]
        y1 = self.agent_pos[1]

        # Conduct scan
        for deg in range(self.sensor_res):
            dI = deg*(np.pi/180) # convert to degree

            # first 90 degrees
            if deg <= 90:
                xNew = -np.sin(dI)*self.sensor_radius + x1  # ccw X
                yNew = np.cos(dI)*self.sensor_radius + y1  # ccw Y

            # 90-180
            if 90 < deg <= 180:
                xNew = -np.cos(dI-90)*self.sensor_radius + x1  # ccw X
                yNew = -np.sin(dI-90)*self.sensor_radius + y1  # ccw Y

            # 180-270
            if 180 < deg <= 270:
                xNew = np.sin(dI-180)*self.sensor_radius + x1  # ccw X
                yNew = -np.cos(dI-180)*self.sensor_radius + y1  # ccw Y

            # 270-360
            if 270 < deg <= 360:
                xNew = np.cos(dI-270)*self.sensor_radius + x1  # ccw X
                yNew = np.sin(dI-270)*self.sensor_radius + y1  # ccw Y

            for w in range(walls.shape[0]):
                # build wall segments
                # format [X3,X4,Y3,Y4]
                wallDict["left"] = [walls[w,0,0], walls[w,2,0], walls[w,0,1], walls[w,2,1]]
                wallDict["right"] = [walls[w,1,0], walls[w,3,0], walls[w,1,1], walls[w,3,1]]
                wallDict["top"] = [walls[w,0,0], walls[w,1,0], walls[w,0,1], walls[w,1,1]]
                wallDict["bottom"] = [walls[w,2,0], walls[w,3,0], walls[w,2,1], walls[w,3,1]]
                
                for key in wallDict:
                    # loop through each part of the rectangle and check if it intersects
                    x3 = wallDict[key][0]+np.random.uniform(low = 0, high=0.1)
                    x4 = wallDict[key][1]+np.random.uniform(low = 0, high=0.1)
                    y3 = wallDict[key][2]+np.random.uniform(low = 0, high=0.1)
                    y4 = wallDict[key][3]+np.random.uniform(low = 0, high=0.1)
                    # direction vector A
                    uAP1 = (x4-x3)*(y1-y3) - (y4-y3)*(x1-x3)
                    uAP2 = (y4-y3)*(xNew-x1) - (x4-x3)*(yNew-y1)
                    uA = np.true_divide(uAP1,uAP2)
                    # direction vector B
                    uBP1 = (xNew-x1)*(y1-y3) - (yNew-y1)*(x1-x3)
                    uBP2 = (y4-y3)*(xNew-x1) - (x4-x3)*(yNew-y1)
                    uB = np.true_divide(uBP1,uBP2)

                    if (0 <= uA <= 1) and (0 <= uB <= 1):
                        # found an intersection, get the distance
                        xIntersect = x1 + (uA*(xNew-x1))
                        yIntersect = y1 + (uA*(yNew-y1))
                        r = np.sqrt((xIntersect-x1) ** 2 + (yIntersect-y1)**2)
                        self.lidar_sensors[deg] = r
                    # leave alone if not intersect and set to inf after checking all lines
            if self.lidar_sensors[deg] == 0:
                self.lidar_sensors[deg] = 3.5
        return self.lidar_sensors

    def agent_step(self, nn_outputs, time_step, walls, world_x, world_y):
        """
        What about negative velocity and theta
        Agent executes movement based on control signals from NN
        Equations from - https://globaljournals.org/GJRE_Volume14/1-Kinematics-Localization-and-Control.pdf
        Blame them if it's wrong.
        :param nn_outputs:
        :param time_step:
        :return:
        """

        [d_vel, d_theta] = nn_outputs  # Change in velocity and theta, respectively
        # scale output in terms of maximum values
        d_vel = d_vel * self.max_vel
        d_theta = d_theta * self.max_rot_vel

        if d_vel > self.max_vel:
            d_vel = self.max_vel
            print("something has gone terribly wrong")

        if d_theta > self.max_rot_vel:
            d_theta = self.max_rot_vel
            print("something has gone horribly wrong")

        # This seems too simple? Like... it has to be more complicated than this... right?
        x_new = self.agent_pos[0] + d_vel * time_step * cos(self.agent_pos[2])
        y_new = self.agent_pos[1] + d_vel * time_step * sin(self.agent_pos[2])
        theta_new = self.agent_pos[2] + d_theta * time_step

        # Boundary checking -- if the new positions are illegal, reject and set collision = true
        # Keep the previous coordinates
        illegal = self.collision_detection(x_new, y_new, walls, world_x, world_y)
        if not illegal:
            self.agent_pos = [x_new, y_new, theta_new]

        return illegal

    def collision_detection(self, x_new, y_new, walls, world_x, world_y):
        """
        This function is called every time step to detect if the agent has run into anything
        Calculates in C-space
        :return: True for collision, false for no collision
        """
        collision = False
        dist = self.body_radius + self.buffer  # Acceptable distance to wall

        if x_new <= 0 + dist or x_new >= world_x - dist:  # Checks outer wall
            collision = True
        elif y_new <= 0 + dist or y_new >= world_y - dist:  # Checks outer wall
            collision = True
        elif walls[0, 0, 0] - dist <= x_new <= walls[0, 3, 0] + dist and \
                walls[0, 3, 1] - dist <= y_new <= walls[0, 0, 1] + dist:  # Checks wall 0
            collision = True
        elif walls[1, 0, 0] - dist <= x_new <= walls[1, 3, 0] + dist and \
                walls[1, 3, 1] - dist <= y_new <= walls[1, 0, 1] + dist:  # Checks wall 1
            collision = True

        return collision

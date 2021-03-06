#!/usr/bin/env python

import numpy as np
from math import sin, cos

class Agent:

    def __init__(self, p, x, y, theta):
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
        self.set_agent_start_pos(x, y, theta)

    def set_agent_start_pos(self, x, y, theta):
        """
        Gives the agent a new starting position in the world (Complete world reset)
        :return:
        """
        self.agent_start_pos = [x, y, theta]
        self.agent_pos = self.agent_start_pos

    def reset_agent_to_start(self):
        """
        Resets agent to its initial position in the world
        :return:
        """
        self.agent_pos = self.agent_start_pos

    def lidar_scan(self, wallDict):
        """
        This function is called when the agent needs to gather LIDAR inputs for the NN
        :return: vector of lidar scan with input size
        """
        # reset lidar each scan
        self.lidar_sensors = np.zeros(self.sensor_res)

        # Current agent position
        agent_x = self.agent_pos[0]
        agent_y = self.agent_pos[1]
        agent_theta = self.agent_pos[2]  # This is in radians
        
        # Conduct scan
        x_new = 0; y_new = 0
        for deg in range(self.sensor_res):
            dI = deg*(np.pi/180)  # convert to radians
            dI += agent_theta

            # first 90 degrees
            if deg <= 90:
                x_new = -np.sin(dI)*self.sensor_radius + agent_x  # ccw X
                y_new = np.cos(dI)*self.sensor_radius + agent_y  # ccw Y

            # 90-180
            if 90 < deg <= 180:
                x_new = -np.cos(dI-(np.pi/2.0))*self.sensor_radius + agent_x  # ccw X
                y_new = -np.sin(dI-(np.pi/2.0))*self.sensor_radius + agent_y  # ccw Y

            # 180-270
            if 180 < deg <= 270:
                x_new = np.sin(dI-np.pi)*self.sensor_radius + agent_x  # ccw X
                y_new = -np.cos(dI-np.pi)*self.sensor_radius + agent_y  # ccw Y

            # 270-360
            if deg > 270:
                x_new = np.cos(dI-(3.0*np.pi/2.0))*self.sensor_radius + agent_x  # ccw X
                y_new = np.sin(dI-(3.0*np.pi/2.0))*self.sensor_radius + agent_y  # ccw Y

            for key in wallDict:
                # http://www.cs.swan.ac.uk/~cssimon/line_intersection.html following this algorithm
                # http://www.jeffreythompson.org/collision-detection/line-rect.php the actual code is from here
                # make the wall/boundary noise by adding in up to 10 cm to the location
                x3 = wallDict[key][0]
                x4 = wallDict[key][1]
                y3 = wallDict[key][2]
                y4 = wallDict[key][3]
                # direction vector A
                uAP1 = ((x4 - x3) * (agent_y - y3)) - ((y4 - y3) * (agent_x - x3))
                uAP2 = ((y4 - y3) * (x_new - agent_x)) - ((x4 - x3) * (y_new - agent_y))
                if uAP2 == 0:  # Prevents division by 0
                    uA = -1
                else:
                    uA = np.true_divide(uAP1, uAP2)
                # direction vector B
                uBP1 = ((x_new - agent_x) * (agent_y - y3)) - ((y_new - agent_y) * (agent_x - x3))
                uBP2 = ((y4 - y3) * (x_new - agent_x)) - ((x4 - x3) * (y_new - agent_y))
                if uBP2 == 0:  # Prevents division by 0
                    uB = -1
                else:
                    uB = np.true_divide(uBP1, uBP2)

                if (0 <= uA <= 1) and (0 <= uB <= 1):
                    # found an intersection, get the distance
                    x_intersect = agent_x + (uA * (x_new - agent_x))
                    y_intersect = agent_y + (uA * (y_new - agent_y))
                    r = np.sqrt((x_intersect - agent_x)**2 + (y_intersect - agent_y)**2)
                    self.lidar_sensors[deg] = r + np.random.normal(0, 0.1)
                # leave alone if not intersect and set to inf after checking all lines
            if self.lidar_sensors[deg] == 0:
                self.lidar_sensors[deg] = 3.5
        return self.lidar_sensors

    def agent_step(self, nn_outputs, time_step, walls, world_x, world_y, threshold):
        """
        What about negative velocity and theta
        Agent executes movement based on control signals from NN
        Equations from - https://globaljournals.org/GJRE_Volume14/1-Kinematics-Localization-and-Control.pdf
        Blame them if it's wrong.
        :param nn_outputs:
        :param time_step:
        :return:
        """

        [d_vel, omega] = nn_outputs  # Change in velocity and theta, respectively
        # scale output in terms of maximum values
        d_vel *= self.max_vel
        omega *= self.max_rot_vel

        if d_vel > self.max_vel:
            d_vel = self.max_vel
            print("something has gone terribly wrong")

        if omega > self.max_rot_vel:
            omega = self.max_rot_vel
            print("something has gone horribly wrong")

        # This seems too simple? Like... it has to be more complicated than this... right?
        x_new = self.agent_pos[0] + d_vel * time_step * cos(self.agent_pos[2])
        y_new = self.agent_pos[1] + d_vel * time_step * sin(self.agent_pos[2])
        theta_new = self.agent_pos[2] + omega * time_step
        if theta_new < 0:
            theta_new += 2 * np.pi
        if theta_new > 2 * np.pi:
            theta_new -= 2 * np.pi

        # Boundary checking -- if the new positions are illegal, reject and set collision = true
        # Keep the previous coordinates
        illegal = self.collision_detection(x_new, y_new, walls, world_x, world_y, threshold)
        if not illegal:
            self.agent_pos = [x_new, y_new, theta_new]

        return illegal

    def collision_detection(self, x_new, y_new, walls, world_x, world_y, threshold):
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
        elif walls[0, 1, 0] - dist <= x_new <= walls[0, 1, 0] + dist and walls[0, 1, 1] - dist <= y_new <= walls[0, 1, 1] + dist:  # Checks wall 0
            collision = True
        elif walls[1, 1, 0] - dist <= x_new <= walls[1, 1, 0] + dist and walls[1, 1, 1] - dist <= y_new <= walls[1, 1, 1] + dist:  # Checks wall 1
            collision = True
        elif walls[0, 1, 0] == walls[1, 1, 0]:  # Checks teleporting when Agent must cross door in X-Direction
            if self.agent_pos[1] <= threshold[0, 1] or self.agent_pos[1] >= threshold[1, 1]:  # Agent not in front of door
                if self.agent_pos[0] <= walls[0, 1, 0] - dist and x_new >= walls[0, 1, 0] - dist:
                    collision = True
                elif self.agent_pos[0] >= walls[1, 1, 0] + dist and x_new <= walls[1, 1, 0] + dist:
                    collision = True
        elif walls[0, 1, 1] == walls[1, 1, 1]:  # Checks teleporting when Agent must cross door in Y-Direction
            if self.agent_pos[0] <= threshold[0, 0] or self.agent_pos[0] >= threshold[1, 0]:  # Agent not in front of door
                if self.agent_pos[1] >= walls[0, 1, 1] + dist and y_new <= walls[0, 1, 1] + dist:
                    collision = True
                elif self.agent_pos[1] <= walls[0, 1, 1] - dist and y_new >= walls[0, 1, 1] - dist:
                    collision = True

        return collision


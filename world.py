#!/usr/bin/env python

import numpy as np

class World:

    def __init__(self, p):
        self.world_x = p.x_dim
        self.world_y = p.y_dim
        self.door_length = p.d_length
        self.door = np.zeros((4, 2))  # (Number of corners, X-Y coordinates)
        self.wall_thickness = p.w_thick
        self.num_walls = p.n_walls
        self.walls = np.zeros((self.num_walls, 4, 2))
        self.collision_penalty = p.coll_penalty
        self.step_penalty = p.stp_penalty

    def world_config1(self):
        """
        Simple world with two walls and one door in the middle of the map
        :return:
        """
        #assert(self.num_walls == 1)

        # Wall 1
        # Top Left Corner
        self.walls[0, 0, 0] = (self.world_x/2)
        self.walls[0, 0, 1] = self.world_y

        # Top Right Corner
        self.walls[0, 1, 0] = self.walls[0, 0, 0] + self.wall_thickness
        self.walls[0, 1, 1] = self.world_y

        # Bottom Left Corner
        self.walls[0, 2, 0] = (self.world_x/2.0)
        self.walls[0, 2, 1] = (self.world_y/2.0) + (self.door_length/2.0)

        # Bottom Right Corner
        self.walls[0, 3, 0] = self.walls[0, 2, 0] + self.wall_thickness
        self.walls[0, 3, 1] = (self.world_y/2.0) + (self.door_length/2.0)

        # Wall 2
        # Top Left Corner
        self.walls[1, 0, 0] = (self.world_x/2)
        self.walls[1, 0, 1] = (self.world_y/2.0) - (self.door_length/2.0)

        # Top Right Corner
        self.walls[1, 1, 0] = self.walls[1, 0, 0] + self.wall_thickness
        self.walls[1, 1, 1] = (self.world_y/2.0) - (self.door_length/2.0)

        # Bottom Left Corner
        self.walls[1, 2, 0] = (self.world_x/2.0)
        self.walls[1, 2, 1] = 0.0

        # Bottom Right Corner
        self.walls[1, 3, 0] = self.walls[1, 2, 0] + self.wall_thickness
        self.walls[1, 3, 1] = 0.0

        # Door
        # Top Left Corner
        self.door[0, 0] = (self.world_x/2)
        self.door[0, 1] = (self.world_y/2) + (self.door_length/2.0)

        # Bottom Left Corner
        self.door[1, 0] = (self.world_x/2)
        self.door[1, 1] = (self.world_y/2) - (self.door_length/2.0)

        # Top Right Corner
        self.door[2, 0] = (self.world_x/2) + self.wall_thickness
        self.door[2, 1] = (self.world_y/2) + (self.door_length/2.0)

        # Bottom Right Corner
        self.door[3, 0] = (self.world_x/2) + self.wall_thickness
        self.door[3, 1] = (self.world_y/2) - (self.door_length/2.0)

    def detect_collision(self, agent_pos, agent_rad):
        """
        Checks the agent position to see if a collision has occurred
        :param agent_pos:
        :param agent_rad:
        :return:
        """

        collision_detected = False

        return collision_detected

    def calculate_reward(self, agent_pos, agent_rad):
        """
        Calculates reward received by agent at each time step
        :return:
        """

        goal = False

        collision = self.detect_collision(agent_pos, agent_rad)

        if collision:
            reward = self.collision_penalty
        elif goal:
            reward = 100.0
        else:
            reward = self.step_penalty

        return reward

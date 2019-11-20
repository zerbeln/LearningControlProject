#!/usr/bin/env python

import numpy as np

class World:

    def __init__(self, p):
        self.world_x = p.x_dim
        self.world_y = p.y_dim
        self.door_length = p.d_length
        self.threshold = np.zeros((2, 2))
        self.wall_thickness = p.w_thick
        self.num_walls = p.n_walls
        self.walls = np.zeros((self.num_walls, 4, 2))
        self.collision_penalty = p.coll_penalty
        self.step_penalty = p.stp_penalty
        self.goal_reward = p.gl.reward
        self.agent_starting_room = 1  # Agent starts in room 1 or room 2

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

        # Define threshold for doors
        # Threshold - Corner 1
        self.threshold[0, 0] = (self.world_x/2)
        self.threshold[0, 1] = (self.world_y/2) + (self.door_length/2.0)

        # Threshold - Corner 2
        self.threshold[1, 0] = (self.world_x/2)
        self.threshold[1, 1] = (self.world_y/2) - (self.door_length/2.0)

    def set_agent_starting_room(self, agent_pos):
        if self.threshold[0, 0] == self.threshold[1, 0]:  # If x values are the same
            if agent_pos[0] < self.threshold[0, 0]:
                self.agent_starting_room = 1
            else:
                self.agent_starting_room = 2

        elif self.threshold[0, 1] == self.threshold[1, 1]:  # If y values are the same
            if agent_pos[1] < self.threshold[0, 1]:
                self.agent_starting_room = 1
            else:
                self.agent_starting_room = 2

    def calculate_reward(self, agent_pos, agent_rad, collision):
        """
        Calculates reward received by agent at each time step
        :return:
        """
        reward = 0
        goal = True

        if collision:
            return self.collision_penalty, False

        if self.agent_starting_room == 1:  # Agent starts in room 1 and crosses to room 2
            if self.threshold[0, 0] == self.threshold[1, 0]:  # Crosses in X-Direction
                if agent_pos[0] + agent_rad < self.threshold[0, 0] or agent_pos[0] - agent_rad < self.threshold[0, 0]:
                    goal = False
            else:  # Crosses in Y-Direction
                if agent_pos[1] + agent_rad < self.threshold[0, 1] or agent_pos[1] - agent_rad < self.threshold[0, 1]:
                    goal = False

        elif self.agent_starting_room == 2:  # Agent starts in room 2 and crosses to room 1
            if self.threshold[0, 0, 0] == self.threshold[0, 1, 0]:  # Crosses in X-Direction
                if agent_pos[0] + agent_rad > self.threshold[0, 0] or agent_pos[0] - agent_rad > self.threshold[0, 0]:
                    goal = False
            else:  # Crosses in Y-Direction
                if agent_pos[1] + agent_rad > self.threshold[0, 1] or agent_pos[1] - agent_rad > self.threshold[0, 1]:
                    goal = False

        if goal:
            return self.goal_reward, goal

        return self.step_penalty, goal

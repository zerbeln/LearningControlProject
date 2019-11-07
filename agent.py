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
        self.max_rot_val = p.max_rot_val
        self.buffer = p.min_dist_to_wall

    def set_agent_start_pos(self):
        """
        Gives the agent a new starting position in the world (Complete world reset)
        :return:
        """

    def reset_agent_to_start(self):
        """
        Resets agent to its initial position in the world
        :return:
        """
        self.agent_pos = self.agent_start_pos.copy()

    def lidar_scan(self, world_x, world_y, walls):
        """
        This function is called when the agent needs to gather inputs for the NN
        :return:
        """

        nn_input_vec = np.zeros(self.n_inputs)

        # Conduct scan
        for deg in range(self.sensor_res):
            dist = 1.0  # Do math to figure out linear distance to wall
            if dist < self.sensor_radius:
                self.lidar_sensors[deg] = dist
            else:
                self.lidar_sensors = math.inf

        # Condense raw sensor data into NN input vector space

        return nn_input_vec

    def agent_step(self, nn_outputs, time_step):
        """
        Agent executes movement based on control signals from NN
        Equations from - https://globaljournals.org/GJRE_Volume14/1-Kinematics-Localization-and-Control.pdf
        Blame them if it's wrong.
        :param nn_outputs:
        :param time_step:
        :return:
        """

        [d_vel, d_theta] = nn_outputs  # Change in velocity and theta, respectively

        if d_vel > self.max_vel:
            d_vel = self.max_vel

        if d_theta > self.max_rot_val:
            d_theta = self.max_rot_val

        # This seems too simple? Like... it has to be more complicated than this... right?
        x_new = self.agent_pos[0] + d_vel * time_step * cos(self.agent_pos[2])
        y_new = self.agent_pos[1] + d_vel * time_step * sin(self.agent_pos[2])
        theta_new = self.agent_pos[2] + d_theta * time_step

        self.agent_pos = [x_new, y_new, theta_new]

    def collision_detection(self):
        """
        This function is called every time step to detect if the agent has run into anything
        :return: True for collision, false for no collision
        """

        if np.amin(self.lidar_sensors) < self.body_radius + self.buffer:
            return True

        return False

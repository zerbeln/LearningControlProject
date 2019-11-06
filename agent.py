import numpy as np
import math

class Agent:

    def __init__(self, p):
        self.sensor_res = p.sensor_resolution
        self.lidar_sensors = np.zeros(self.sensor_res)
        self.agent_pos = np.zeros(2)
        self.agent_start_pos = np.zeros(2)
        self.n_inputs = p.num_inputs
        self.body_radius = p.agent_rad
        self.sensor_radius = p.detection_radius

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

    def agent_step(self, nn_outputs):
        """
        Agent executes movement based on control signals from NN
        :param nn_outputs:
        :return:
        """
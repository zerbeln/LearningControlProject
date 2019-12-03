#!/usr/bin/env python

import rospy

# The velocity command message
from geometry_msgs.msg import Twist

# The laser scan message
from sensor_msgs.msg import LaserScan
import neural_network as neu_net
from parameters import Parameters as p
import numpy as np

class GoForward():
    def __init__(self):
        # initiliaze
        rospy.init_node('ILikeToMoveIt', anonymous=False)

        # tell user how to stop TurtleBot
        rospy.loginfo("To stop TurtleBot CTRL + C")

        # What function to call when you ctrl + c
        rospy.on_shutdown(self.shutdown)

        # Create a publisher which can "talk" to TurtleBot and tell it to move
        # Tip: You may need to change cmd_vel_mux/input/navi to /cmd_vel if you're not using TurtleBot2
        self.cmd_vel = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

        # A subscriber for the laser scan data
        self.laser_sub = rospy.Subscriber('scan', LaserScan, self.call_nn)

        # TurtleBot will stop if we don't keep telling it to move.  How often should we tell it to move? 10 HZ
        r = rospy.Rate(10)

        # Turtlebot3 Burger specs
        self.max_vel = 0.22  # m/s
        self.max_rot_vel = 0.5  # rad/s

        # Twist is a datatype for velocity
        self.move_cmd = Twist()
        self.move_cmd.linear.x = 0.0
        self.move_cmd.angular.z = 1.5

        self.nn = neu_net.NeuralNetwork(p)
        self.nn.get_nn_weights_from_file('BestNN.csv')

        # as long as you haven't ctrl + c keeping doing...
        while not rospy.is_shutdown():
            # publish the velocity
            self.cmd_vel.publish(self.move_cmd)
            # wait for 0.1 seconds (10 HZ) and publish again
            r.sleep()

    def call_nn(self, msg):
        """Gets called every time a laser scan is received"""

        laser_msg = np.zeros(360)

        for i in range(len(msg.ranges)):
            if msg.ranges[i] > 3.5:
                laser_msg[i] = 3.5
            else:
                laser_msg[i] = msg.ranges[i]

        # reverse_arr = laser_msg[::-1]
        self.nn.get_outputs(laser_msg)
        [d_vel, omega] = self.nn.out_layer  # Change in velocity and theta, respectively

        # scale output in terms of maximum values
        d_vel *= self.max_vel
        omega *= self.max_rot_vel

        self.move_cmd.linear.x = -d_vel
        self.move_cmd.angular.z = omega
        print(self.move_cmd.linear.x, self.move_cmd.angular.z)

    def shutdown(self):
        # stop turtlebot
        rospy.loginfo("Stop TurtleBot")
        # a default Twist has linear.x of 0 and angular.z of 0.  So it'll stop TurtleBot
        self.cmd_vel.publish(Twist())
        # sleep just makes sure TurtleBot receives the stop command prior to shutting down the script
        # rospy.sleep(1)
        exit()


if __name__ == '__main__':

    try:
        GoForward()
    except:
        rospy.loginfo("ILikeToMoveIt node terminated.")

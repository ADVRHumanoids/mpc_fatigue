#!/usr/bin/env python
import rospy
import numpy as np
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import Header
 
def callback(data):
    #Once you receive the trajectory:
    trajectory.append(np.array(data.data))
    settato = 1 

#def interpolate():
#    #The interpolate function, starting from two optimal trajectory points and current time, will generate the current state according to the rospy.Rate
#    N = 40
#    T = 30.0
#    h = T / N
#    cu


if __name__ == '__main__':
    #Define the node that reads the OPTIMAL TRAJECTORY and sned the joint state position while running
    rospy.init_node('unroller_node')
    #Define the publisher
    pub = rospy.Publisher("to_robot_topic", String, queue_size=10)
    #Define the subscriber
    rospy.Subscriber("to_unroller_topic", Float32MultiArray, callback)
    #Define the frequency at which the unroller will send information to the robot
    rate = rospy.Rate(100) # 10hz
    rospy.spin()
    trajectory = []
    settato = 0
    while settato:
        print(trajectory[0])
        print(trajectory[0][0])
    
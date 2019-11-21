#!/usr/bin/env python


import rospy
from sensor_msgs.msg import JointState
from std_msgs.msg import Header
from Centauros_features import *



def talker(q1):
    #Define publisher
    pub = rospy.Publisher('topic_position_from_invkin', JointState, queue_size=10) # This can be seen in rostopic list
    rospy.init_node('node_position_from_invkin') #This can be seen in rosnode list
    hello_str = JointState()
    hello_str.header = Header()
    hello_str.header.stamp = rospy.Time.now()
    hello_str.name = Centauro_features.joint_name
    hello_str.position =  []
    hello_str.velocity = []
    hello_str.effort = []
    rate = rospy.Rate(1000) # 10hz
    i = 0.0

    while not rospy.is_shutdown():
        pub.publish(hello_str)
        rate.sleep()
        hello_str.header.stamp = rospy.Time.now()
        hello_str.position =  [i/1000*q1[0],i/1000*q1[1],i/1000*q1[2],i/1000*q1[3],i/1000*q1[4],i/1000*q1[5],i/1000*q1[6],i/1000*q1[7],i/1000*q1[8],i/1000*q1[9],i/1000*q1[10],i/1000*q1[11],i/1000*q1[12],i/1000*q1[13]]
        i+=1.0
        if i == 1000:
            break
        

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass

#!/usr/bin/env python
# Software License Agreement (BSD License)
#
# Copyright (c) 2008, Willow Garage, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#  * Neither the name of Willow Garage, Inc. nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# Revision $Id

## Simple talker demo that published std_msgs/Strings messages
## to the 'chatter' topic

from casadi import *
import rospy
import roslib; roslib.load_manifest('visualization_marker_tutorials')
from sensor_msgs.msg import JointState
from std_msgs.msg import Header
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
import mpc_fatigue.pynocchio_casadi as pin
import numpy as np
import math




def talker(q1,q2,F,box):

    #Load URDF
    pilz_first_urdf = rospy.get_param('/robot_description_1')
    pilz_second_urdf = rospy.get_param('/robot_description_2')

    # ---------------------- Solve forward kinematics pilz 1  ---------------------
    #Postion of link 5 in cartesian space (Solution to direct kinematics)
    fk_first_string = pin.generate_forward_kin(pilz_first_urdf, 'prbt_link_5')
    #Create casadi function
    fk_first = casadi.Function.deserialize(fk_first_string)

    # ---------------------- Solve forward kinematics pilz 2  ---------------------
    #Postion of link 5 in cartesian space (Solution to direct kinematics)
    fk_second_string = pin.generate_forward_kin(pilz_second_urdf, 'prbt_link_5')
    #Create casadi function
    fk_second = casadi.Function.deserialize(fk_second_string)

    #Define the talker parameters
    if True: 
        pub = rospy.Publisher('topic_position_from_invkin', JointState, queue_size=100) # This can be seen in rostopic list
        rospy.init_node('node_position_from_invkin') #This can be seen in rosnode list
        hello_str = JointState()
        hello_str.header = Header()
        hello_str.header.stamp = rospy.Time.now()
        hello_str.name = ['prbt_joint_1','prbt_joint_2','prbt_joint_3','prbt_joint_4','prbt_joint_5','prbt_joint_6','sec_prbt_joint_1','sec_prbt_joint_2','sec_prbt_joint_3','sec_prbt_joint_4','sec_prbt_joint_5','sec_prbt_joint_6']
        hello_str.position =  []
        hello_str.velocity = []
        hello_str.effort = []
        rate = rospy.Rate(10) # 10hz
        i = 0
        qsize = np.size(q1[0])
        fsize = np.size(F[0])

    #Define the marker plotter
    if True:
        
        topic = 'visualization_marker_array'
        publisher_1 = rospy.Publisher(topic, MarkerArray,queue_size=100)
        publisher_2 = rospy.Publisher(topic, MarkerArray,queue_size=100)
        publisher_3 = rospy.Publisher(topic, MarkerArray,queue_size=100)
        #rospy.init_node('register')
        markerArray = MarkerArray()
    
        marker_f = Marker()
        marker_s = Marker()
        weight = Marker()
        marker_f.header.frame_id = "/prbt_base"
        marker_s.header.frame_id = "/prbt_base"
        weight.header.frame_id = "/prbt_base"
        marker_f.ns = "Fz_forces";
        marker_s.ns = "Fz_forces";
        weight.ns = "Fz_forces";
        marker_f.type = marker_f.ARROW
        marker_s.type = marker_s.ARROW
        weight.type = marker_s.ARROW
        marker_f.action = marker_f.ADD
        marker_s.action = marker_s.ADD
        weight.action = weight.ADD
        marker_f.id = 0;
        marker_s.id = 1;
        weight.id = 2;
        #Scales
        marker_f.scale.x = 0.2
        marker_s.scale.x = 0.2
        weight.scale.x = 0.2

        marker_f.scale.y = 0.02
        marker_s.scale.y = 0.02
        weight.scale.y = 0.02

        marker_f.scale.z = 0.02
        marker_s.scale.z = 0.02
        weight.scale.z = 0.02

        marker_f.color.a = 1.0
        marker_f.color.r = 1.0
        marker_f.color.g = 1.0
        marker_f.color.b = 1.0

        marker_s.color.a = 1.0
        marker_s.color.r = 1.0
        marker_s.color.g = 1.0
        marker_s.color.b = 1.0

        weight.color.a = 1.0
        weight.color.r = 1.0
        weight.color.g = 1.0
        weight.color.b = 1.0


        marker_f.pose.orientation.x = 0.0
        marker_f.pose.orientation.y = 1.0
        marker_f.pose.orientation.z = 0.0
        marker_f.pose.orientation.w = -1.0

        marker_s.pose.orientation.x = 0.0
        marker_s.pose.orientation.y = 1.0
        marker_s.pose.orientation.z = 0.0
        marker_s.pose.orientation.w = -1.0

        weight.pose.orientation.x = 0.0
        weight.pose.orientation.y = 1.0
        weight.pose.orientation.z = 0.0
        weight.pose.orientation.w = 1.0

    while not rospy.is_shutdown():
        if i < qsize:

            if i < fsize:
                qf = [q1[0][i],q1[1][i],q1[2][i],q1[3][i],q1[4][i],q1[5][i]]
                qs = [q2[0][i],q2[1][i],q2[2][i],q2[3][i],q2[4][i],q2[5][i]]
                pos_Fzf = fk_first(q = qf)['ee_pos']
                pos_Fzs = fk_second(q = qs)['ee_pos']
                #Marker scales
                marker_f.scale.x = 0.2*F[0][i]/F[2]
                marker_s.scale.x = 0.2*F[1][i]/F[2]
                weight.scale.x = 0.2*(F[1][i] + F[0][i]) /F[2]

                # Marker f position 
                marker_f.pose.position.x = pos_Fzf[0]+0.1
                marker_f.pose.position.y = pos_Fzf[1]
                marker_f.pose.position.z = pos_Fzf[2]

                marker_s.pose.position.x = pos_Fzs[0]-0.1
                marker_s.pose.position.y = pos_Fzs[1]
                marker_s.pose.position.z = pos_Fzs[2]

                weight.pose.position.x = box[0][i]
                weight.pose.position.y = box[1][i]
                weight.pose.position.z = box[2][i]


            pub.publish(hello_str)
            publisher_1.publish([marker_f])
            publisher_2.publish([marker_s])
            publisher_3.publish([weight])
            rate.sleep()
            hello_str.header.stamp = rospy.Time.now()
            hello_str.position = [q1[0][i],q1[1][i],q1[2][i],q1[3][i],q1[4][i],q1[5][i],q2[0][i],q2[1][i],q2[2][i],q2[3][i],q2[4][i],q2[5][i]] 
            
        else:
            marker_f.action = marker_f.DELETE
            marker_s.action = marker_s.DELETE
            weight.action = weight.DELETE
            publisher_1.publish([marker_f])
            publisher_2.publish([marker_s])
            publisher_3.publish([weight])
            break
        i += 1
        

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass

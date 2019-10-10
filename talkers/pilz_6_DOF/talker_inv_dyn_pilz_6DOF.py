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

import rospy
from sensor_msgs.msg import JointState
from std_msgs.msg import Header
import numpy as np



def talker(q1,q2,q3,q4,q5,q6):
    pub = rospy.Publisher('topic_position_from_invkin', JointState, queue_size=10) # This can be seen in rostopic list
    rospy.init_node('node_position_from_invkin') #This can be seen in rosnode list
    hello_str = JointState()
    hello_str.header = Header()
    hello_str.header.stamp = rospy.Time.now()
    hello_str.name = ['prbt_joint_1','prbt_joint_2','prbt_joint_3','prbt_joint_4','prbt_joint_5','prbt_joint_6']
    hello_str.position =  []
    hello_str.velocity = []
    hello_str.effort = []
    rate = rospy.Rate(10) # 10hz
    i = 0
    qsize = np.size(q1)

    while not rospy.is_shutdown():
        if i < qsize:
            #for k in range(rep):
            pub.publish(hello_str)
            rate.sleep()
            hello_str.header.stamp = rospy.Time.now()
            hello_str.position = [q1[i],q2[i],q3[i],q4[i],q5[i],q6[i]]  
        else:
            break
        i += 1
        

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass

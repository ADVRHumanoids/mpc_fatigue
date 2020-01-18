#!/usr/bin/env python
import rospy
import numpy as np
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32MultiArray
from visualization_msgs.msg import Marker
from geometry_msgs.msg import WrenchStamped
from std_msgs.msg import Header
import time

def callback(data):
    #Construct the time grid
    if tgrid:
        tgrid.append((tgrid[-1] + h))
    else:
        tgrid.append(0.0)
#    #Once you receive the trajectory:
    data = np.array(data.data)
    force.append(data[14:20])
    trajectory.append(data[0:14]) 
    
    
def define_marker(marker):
    #Marker properties
    marker.header.frame_id = "mass1_ee";
    marker.type = 1 # Number one means CUBE
    marker.action = 0
    marker.pose.position.x = 0.0
    marker.pose.position.y = - 0.21
    marker.pose.position.z = 0.0
    marker.pose.orientation.x = 0
    marker.pose.orientation.y = 1
    marker.pose.orientation.z = 0
    marker.pose.orientation.w = 1
    marker.scale.x = 0.15
    marker.scale.y = 0.35
    marker.scale.z = 0.15
    marker.color.a = 1.0
    marker.color.r = 0.0
    marker.color.g = 1.0
    marker.color.b = 0.0
    
def define_joint(joint_angle):
    joint_angle.header = Header()
    joint_angle.header.stamp = rospy.Time.now()
    joint_angle.name = ['j_arm1_1','j_arm1_2','j_arm1_3','j_arm1_4','j_arm1_5','j_arm1_6','j_arm1_7','j_arm2_1','j_arm2_2','j_arm2_3','j_arm2_4','j_arm2_5','j_arm2_6','j_arm2_7']
    joint_angle.position =  []
    joint_angle.velocity = []
    joint_angle.effort = []
    
def define_wrench(RightWrench,LeftWrench):
    RightWrench.header.frame_id = "mass2_ee"
    LeftWrench.header.frame_id = "mass1_ee"
    
if __name__ == '__main__':
    #Initialization

    N = 40
    T = 30.0
    h = T / N
    #Empty list
    trajectory = []
    force = []
    tgrid = []
    
    #Define here how fast you want publish data to the robot
    publish_rate = 120
    
    #Define the node that reads the OPTIMAL TRAJECTORY and sned the joint state position while running
    rospy.init_node('unroller_node')
    #Define the subscriber
    rospy.Subscriber("to_unroller_topic", Float32MultiArray,callback)
    #Define the joint state publisher
    pubs = rospy.Publisher("to_robot_topic",  JointState, queue_size=100)
    #Define the marker publisher
    marker_pub = rospy.Publisher('MarkerTopic', Marker, queue_size=10)
    #Define the wrench publisher
    wrench_pub = rospy.Publisher('WrenchStampedTopic', WrenchStamped, queue_size=10)
    #Define the frequency at which the unroller will send information to the robot
    rates = rospy.Rate(publish_rate) # 10hz
    

    #Definition of JointState message
    joint_angle = JointState()
    define_joint(joint_angle)
    
    #Definition of Marker message
    marker = Marker()  
    define_marker(marker)
    
    #Definition of WrenchStamped
    RightWrench = WrenchStamped()
    LeftWrench = WrenchStamped()
    define_wrench(RightWrench,LeftWrench)
    
    while True:
#        print("time" + str(time.time()))
#        print("trajectory array :=" + str(trajectory))
#        print("tgrid array:= " + str(tgrid))
#        print("force array:= " + str(force))
#        print("Trajectory length:" + str(np.size(trajectory)))
#        print("tgrid length:" + str(np.size(tgrid)))# + "Time: " + str(tgrid[-1]))

        if trajectory:
            #Define time zero
            temp_zero = time.time()
            
            #Show the trajectory in Rviz
            joint_angle.header.stamp = rospy.Time.now()
            joint_angle.position = trajectory[0]
            pubs.publish(joint_angle)

            #Show the box
            marker_pub.publish(marker)
            
            #Extract initial condition 
            p0 = trajectory.pop(0)
            t0 = tgrid.pop(0)
            f0 = force.pop(0)
            
            #Show the forces
            LeftWrench.wrench.force.x = f0[0] / 9.81 * 10
            LeftWrench.wrench.torque.y = f0[1] / 9.81 * 10
            LeftWrench.wrench.force.z = f0[2] / 9.81 * 10
            RightWrench.wrench.force.x = f0[3] / 9.81 * 10
            RightWrench.wrench.torque.y = f0 [4] / 9.81 * 10
            RightWrench.wrench.force.z = f0[5] / 9.81 * 10
            wrench_pub.publish(RightWrench)
            wrench_pub.publish(LeftWrench) 
            
            rates.sleep()
            
            #Comincio ad interpolare.
            while len(trajectory)-1:
                
                print(temp_zero)
                #Evaluate how much time has passed since I started sending the trajectory
                tnow = time.time() - temp_zero
                print("Simulation time: " + str(tnow))

                try:
                    if tnow > tgrid[0]:
                        p0 = trajectory.pop(0)
                        f0 = force.pop(0)
                        t0 = tgrid.pop(0)
                except:
                    #print("Trajectory is empty : MPC_termination")
                    break
    
                #Evaluate interpolated joint angle values
                joint_pos = p0 + (trajectory[0] - p0)/h * (tnow - t0)
                Fopt = f0 + (force[0] - f0)/h * (tnow - t0)
                
                #Send data to the robot
                joint_angle.header.stamp = rospy.Time.now()
                joint_angle.position = joint_pos
                pubs.publish(joint_angle)
                
                marker_pub.publish(marker)

                LeftWrench.wrench.force.x = Fopt[0] / 9.81 * 10
                LeftWrench.wrench.force.y = Fopt[1] / 9.81 * 10
                LeftWrench.wrench.force.z = Fopt[2] / 9.81 * 10
                RightWrench.wrench.force.x = Fopt[3] / 9.81 * 10
                RightWrench.wrench.force.y = Fopt[4] / 9.81 * 10
                RightWrench.wrench.force.z = Fopt[5] / 9.81 * 10
                wrench_pub.publish(RightWrench)
                wrench_pub.publish(LeftWrench)  
                
  
                rates.sleep()
            
            
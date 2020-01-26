#!/usr/bin/env python
import rospy
import numpy as np
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32MultiArray
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from geometry_msgs.msg import WrenchStamped
from std_msgs.msg import Header
import time
from MPC_parameters import *

def callback(data):
    #Construct the time grid
    if tgrid:
        tgrid.append((tgrid[-1] + h))
    else:
        tgrid.append(0.0)
#    #Once you receive the trajectory:
    data = np.array(data.data)
    force.append(data[14:20])
    torque.append(data[20:34])
    temperature.append(data[34:48])
    trajectory.append(data[0:14]) 
    
    
def define_box_marker(marker):
    #Marker properties
    marker.header.frame_id = "mass1_ee";
    marker.type = 1 # Number one means CUBE
    marker.action = 0
    marker.pose.position.x = 0.0
    marker.pose.position.y = - mpc.Lbox/2
    marker.pose.position.z = 0.0
    marker.pose.orientation.x = 0
    marker.pose.orientation.y = 1
    marker.pose.orientation.z = 0
    marker.pose.orientation.w = 1
    marker.scale.x = 0.15
    marker.scale.y = mpc.Lbox
    marker.scale.z = 0.15
    marker.color.a = 0.5
    marker.color.r = 0.0
    marker.color.g = 1.0
    marker.color.b = 0.0
    marker.ns = "Box"
    
    
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

def define_temperature_marker(marker,index,Temperature):
    marker.header.frame_id = str(joint_angle.name[index][2:]);
    marker.id = index
    marker.type = 2 # Number one means CUBE
    marker.action = 0
    marker.pose.position.x = 0.0
    marker.pose.position.y =  0.0
    marker.pose.position.z = 0.0
    marker.pose.orientation.x = 0
    marker.pose.orientation.y = 1
    marker.pose.orientation.z = 0
    marker.pose.orientation.w = 1
    marker.scale.x = Temperature[index]/(mpc.temperature_bound*10)
    marker.scale.y = Temperature[index]/(mpc.temperature_bound*10)
    marker.scale.z = Temperature[index]/(mpc.temperature_bound*10)
    marker.color.a = 0.5
    marker.color.r = 0.0
    marker.color.g = 1.0
    marker.color.b = 0.0
    marker.ns = "Temperature"
    
    
if __name__ == '__main__':
    #Initialization

    N = mpc.N
    T = mpc.T
    h = T / N
    #Empty list
    trajectory = []
    torque = []
    force = []
    temperature = []
    tgrid = []
    
    #Define the node that reads the OPTIMAL TRAJECTORY and sned the joint state position while running
    rospy.init_node('unroller_node')
    #Define the subscriber
    rospy.Subscriber("to_unroller_topic", Float32MultiArray,callback)
    #Define the joint state publisher
    pubs = rospy.Publisher("to_robot_topic",  JointState, queue_size=100)
    #Define the box marker publisher
    Box_marker_pub = rospy.Publisher('BoxMarkerTopic', Marker, queue_size=10)
    #Define the temperature marker publisher
    Temp_marker_pub = rospy.Publisher('TemperatureMarkerTopic', MarkerArray, queue_size=10)
    #Define the wrench publisher
    wrench_pub = rospy.Publisher('WrenchStampedTopic', WrenchStamped, queue_size=10)
    #Define the frequency at which the unroller will send information to the robot
    rates = rospy.Rate(mpc.unroller_publish_rate) # 10hz
    

    #Definition of JointState message
    joint_angle = JointState()
    define_joint(joint_angle)
    
    #Definition of Marker message
    Boxmarker = Marker()  
    define_box_marker(Boxmarker)
    #Definition of the temperature marker
    TmarkerArray = MarkerArray()
    
    #Definition of WrenchStamped
    RightWrench = WrenchStamped()
    LeftWrench = WrenchStamped()
    define_wrench(RightWrench,LeftWrench)
    
    while True:
        if trajectory:
            #Define time zero
            temp_zero = time.time()
            
            #Show the trajectory in Rviz
            joint_angle.header.stamp = rospy.Time.now()
            joint_angle.position = trajectory[0]
            pubs.publish(joint_angle)

            #Show the box
            Box_marker_pub.publish(Boxmarker)
            
            #Extract initial condition 
            p0 = trajectory.pop(0)
            t0 = tgrid.pop(0)
            f0 = force.pop(0)
            tau0 = torque.pop(0)
            Temp0 = temperature.pop(0)
            
            #Show the forces
            LeftWrench.wrench.force.x =  f0[0] / 9.81 * mpc.m
            LeftWrench.wrench.force.y =  f0[1] / 9.81 * mpc.m
            LeftWrench.wrench.force.z =  f0[2] / 9.81 * mpc.m
            RightWrench.wrench.force.x =  f0[3] / 9.81 * mpc.m
            RightWrench.wrench.force.y =  f0 [4] / 9.81 * mpc.m
            RightWrench.wrench.force.z =  f0[5] / 9.81 * mpc.m
            
            wrench_pub.publish(RightWrench)
            wrench_pub.publish(LeftWrench) 
            
            for index in range(0,mpc.nq):
                    Tmarker = Marker()
                    define_temperature_marker(Tmarker,index,Temp0)
                    TmarkerArray.markers.append(Tmarker)
            
            Temp_marker_pub.publish(TmarkerArray)
            TmarkerArray.markers = []
            rates.sleep()
            
            #Comincio ad interpolare.
            while len(trajectory)-1:
                
                #print(temp_zero)
                #Evaluate how much time has passed since I started sending the trajectory
                tnow = time.time() - temp_zero
                print("Simulation time: " + str(tnow))

                try:
                    if tnow > tgrid[0]:
                        f0 = force.pop(0)
                        t0 = tgrid.pop(0)
                        tau0 = torque.pop(0)
                        p0 = trajectory.pop(0)
                        Temp0 = temperature.pop(0)
                except:
                    #print("Trajectory is empty : MPC_termination")
                    break
    
                #Evaluate interpolated joint angle values
                joint_pos = p0 + (trajectory[0] - p0)/h * (tnow - t0)
                Fopt = f0 + (force[0] - f0)/h * (tnow - t0)
                Tau_opt = tau0 + (torque[0] - tau0)/h * (tnow - t0)
                Temp_opt = Temp0 + (temperature[0] - Temp0)/h * (tnow - t0)
                
                #Send data to the robot
                joint_angle.header.stamp = rospy.Time.now()
                joint_angle.position = joint_pos
                joint_angle.effort = Tau_opt.tolist()
                
                LeftWrench.wrench.force.x =  Fopt[0] / 9.81 * mpc.m
                LeftWrench.wrench.force.y =  Fopt[1] / 9.81 * mpc.m
                LeftWrench.wrench.force.z =  Fopt[2] / 9.81 * mpc.m
                RightWrench.wrench.force.x =  Fopt[3] / 9.81 * mpc.m
                RightWrench.wrench.force.y =  Fopt[4] / 9.81 * mpc.m
                RightWrench.wrench.force.z =  Fopt[5] / 9.81 * mpc.m
                
                for index in range(0,mpc.nq):
                    Tmarker = Marker()
                    define_temperature_marker(Tmarker,index,Temp0)
                    TmarkerArray.markers.append(Tmarker)
            
                Temp_marker_pub.publish(TmarkerArray)
                pubs.publish(joint_angle)
                Box_marker_pub.publish(Boxmarker)
                wrench_pub.publish(RightWrench)
                wrench_pub.publish(LeftWrench)  
                
                TmarkerArray.markers = []
                rates.sleep()
            
            
#!/usr/bin/env python
import mpc_fatigue.pynocchio_casadi as pin
from MPC_parameters import *
from casadi import *
import numpy as np
import rospy
import csv
from visualization_msgs.msg import Marker
from sensor_msgs.msg import JointState
from std_msgs.msg import Header


#ROS parameters
centauro = rospy.get_param('/robot_description')
end_LA = 'mass1_ee'
end_RA = 'mass2_ee'
# General parameters
nq = mpc.nq


class Centauro_features:
    
    #Define the joints that will be controlled by the python script
    joint_name = ['j_arm1_1','j_arm1_2','j_arm1_3','j_arm1_4','j_arm1_5','j_arm1_6','j_arm1_7','j_arm2_1','j_arm2_2','j_arm2_3','j_arm2_4','j_arm2_5','j_arm2_6','j_arm2_7']

    #Define joint position lower bound
    joint_angle_lb = np.array([-3.312,0.020,-2.552,-2.465,-2.569,-1.529,-2.565,-3.3458,-3.4258,-2.5614,-2.4794,-2.5394,-1.5154,-2.5554])

    #Define joint position upper bound
    joint_angle_ub = np.array([1.615,3.431,2.566,0.280,2.562,1.509,2.569,1.6012,-0.0138,2.5606,0.2886,2.5546,1.5156,2.5686])

    #Define joint velocity limit
    joint_velocity_lim = np.array([3.86,3.86,6.06,6.06,11.72,11.72,20.35,3.86,3.86,6.06,6.06,11.72,11.72,20.35])

    #Define torque limit
    joint_torque_lim = np.array([147.00,147.00,147.00,147.00,55.00,55.00,28.32,147.00,147.00,147.00,147.00,55.00,55.00,28.32])

    # Compute forward kinematics
    fk_la = casadi.Function.deserialize(pin.generate_forward_kin(centauro, end_LA))
    fk_ra = casadi.Function.deserialize(pin.generate_forward_kin(centauro, end_RA))

    # Jacobians
    jac_la = casadi.Function.deserialize(pin.generate_jacobian(centauro, end_LA))
    jac_ra = casadi.Function.deserialize(pin.generate_jacobian(centauro, end_RA))

    # Inverse dynamics
    Idyn = casadi.Function.deserialize(pin.generate_inv_dyn(centauro))
   
             
#Returns position, rotation or all information of the left arm
def ForwKinLA(qc,typ = "all"):
        if typ == "pos":
            return Centauro_features.fk_la(q = qc)['ee_pos']
        elif typ == "rot":
            return Centauro_features.fk_la(q = qc)['ee_rot']
        else:
            return Centauro_features.fk_la(q = qc)

#Returns position, rotation or all information of the right arm
def ForwKinRA(qc,typ = "all"):
        if typ == "pos":
            return Centauro_features.fk_ra(q = qc)['ee_pos']
        elif typ == "rot":
            return Centauro_features.fk_ra(q = qc)['ee_rot']
        else:
            return Centauro_features.fk_ra(q = qc)

#Returns Jacobian of left arm
def Jac_LA(qc):
    return Centauro_features.jac_la(q = qc)["J"][:,0:7]

#Returns Jacobian of left arm
def Jac_RA(qc):
    return Centauro_features.jac_ra(q = qc)["J"][:,7:]

#Returns Inverse dynamics of the robot
def InvDyn(qc,qcdot,qcddot):
    return Centauro_features.Idyn(q = qc, qdot = qcdot, qddot = qcddot)['tau']

#Compute quatersions starting from the rotation matrix
def CompQuat(rot):
    return  np.array([sqrt(rot[0,0] + rot[1,1] + rot[2,2] + 1),
                          np.sign(rot[2,1] - rot[1,2]) * sqrt(rot[0,0] - rot[1,1] - rot[2,2] + 1),
                          np.sign(rot[0,2] - rot[2,0]) * sqrt(rot[1,1] - rot[2,2] - rot[0,0] + 1),
                          np.sign(rot[1,0] - rot[0,1])*  sqrt(rot[2,2] - rot[0,0] - rot[1,1] + 1)])

#Compute roll pitch yaw starting from the rotation matrix
def CompRPY(rot):
    Roll = np.arctan2(rot[2,0],rot[2,1])
    Pitch = np.arccos( -rot[2,2])
    Yaw = - np.arctan2(rot[0,2],rot[1,2])
    return([Roll,Pitch,Yaw])

#Save csv
def SaveCsv(folder,qc_opt,qcd_opt,F_opt_glob, F_opt_loc,tau_LR,tau_RR,lbt,ubt,box_pos,box_rpy,Tw,lbtemp,ubtemp):

    #Save the optimal joint angles
    with open('/home/user/workspace/src/mpc_fatigue/Centauro_solutions/' + folder + '/qc_optimal.csv', 'wb') as myfile:
        wr = csv.writer(myfile)
        wr.writerow(qc_opt)
    
    #Save the optimal joint velocities
    with open('/home/user/workspace/src/mpc_fatigue/Centauro_solutions/' + folder + '/qcdot_optimal.csv', 'wb') as myfile:
        wr = csv.writer(myfile)
        wr.writerow(qcd_opt)
    
    #Save the optimal global force at the two end effector
    with open('/home/user/workspace/src/mpc_fatigue/Centauro_solutions/' + folder + '/F_optimal_glob.csv', 'wb') as myfile:
        wr = csv.writer(myfile)
        wr.writerow(F_opt_glob)
        
    #Save the optimal local force at the two end effector
    with open('/home/user/workspace/src/mpc_fatigue/Centauro_solutions/' + folder + '/F_optimal_loc.csv', 'wb') as myfile:
        wr = csv.writer(myfile)
        wr.writerow(F_opt_loc)
    
    with open('/home/user/workspace/src/mpc_fatigue/Centauro_solutions/' + folder + '/tauL_optimal.csv', 'wb') as myfile:
        wr = csv.writer(myfile)
        wr.writerow(tau_LR)

    with open('/home/user/workspace/src/mpc_fatigue/Centauro_solutions/' + folder + '/tauR_optimal.csv', 'wb') as myfile:
        wr = csv.writer(myfile)
        wr.writerow(tau_RR)
        
    with open('/home/user/workspace/src/mpc_fatigue/Centauro_solutions/'+ folder +'/lbt.csv', 'wb') as myfile:
        wr = csv.writer(myfile)
        wr.writerow(lbt)
        
    with open('/home/user/workspace/src/mpc_fatigue/Centauro_solutions/' +folder + '/ubt.csv', 'wb') as myfile:
        wr = csv.writer(myfile)
        wr.writerow(ubt)

    with open('/home/user/workspace/src/mpc_fatigue/Centauro_solutions/'+ folder +'/box_pos.csv', 'wb') as myfile:
        wr = csv.writer(myfile)
        wr.writerow(box_pos)
        
    with open('/home/user/workspace/src/mpc_fatigue/Centauro_solutions/' +folder + '/box_rpy.csv', 'wb') as myfile:
        wr = csv.writer(myfile)
        wr.writerow(box_rpy)

    with open('/home/user/workspace/src/mpc_fatigue/Centauro_solutions/' + folder + '/Tw.csv', 'wb') as myfile:
        wr = csv.writer(myfile)
        wr.writerow(Tw)

    with open('/home/user/workspace/src/mpc_fatigue/Centauro_solutions/' + folder + '/lbtemp.csv', 'wb') as myfile:
        wr = csv.writer(myfile)
        wr.writerow(lbtemp)

    with open('/home/user/workspace/src/mpc_fatigue/Centauro_solutions/' + folder + '/ubtemp.csv', 'wb') as myfile:
        wr = csv.writer(myfile)
        wr.writerow(ubtemp)

def InvTalker(q1):
        #Define publisher
        pub = rospy.Publisher('to_robot_topic', JointState, queue_size=10) # This can be seen in rostopic list
        joint = JointState()
        joint.header = Header()
        joint.header.stamp = rospy.Time.now()
        joint.name = Centauro_features.joint_name
        joint.position =  []
        joint.velocity = []
        joint.effort = []
        rate = rospy.Rate(mpc.initial_position_rate) # 10hz
        i = 0.0

        #Marker properties
        marker = Marker()  
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
        marker_pub = rospy.Publisher('MarkerTopic', Marker, queue_size=10)
        #marker_rate = rospy.Rate(1000) # 10hz

        while not rospy.is_shutdown():
            pub.publish(joint)
            marker_pub.publish(marker)
            rate.sleep()
            joint.header.stamp = rospy.Time.now()
            marker.header.stamp = rospy.Time.now()
            joint.position =  [i/1000*q1[0],i/1000*q1[1],i/1000*q1[2],i/1000*q1[3],i/1000*q1[4],i/1000*q1[5],i/1000*q1[6],i/1000*q1[7],i/1000*q1[8],i/1000*q1[9],i/1000*q1[10],i/1000*q1[11],i/1000*q1[12],i/1000*q1[13]]
            i+=1.0
            if i == mpc.initial_position_rate:
                break


#From the desider initial condition, compute the inverse kinematic to determine
def InvKin(BoxPos,L):
    w = []
    lbw = []
    ubw = []

    qc = SX.sym('qc', nq)
    w.append(qc)
    lbw = vertcat(Centauro_features.joint_angle_lb)
    ubw = vertcat(Centauro_features.joint_angle_ub)

    pos_LA = ForwKinLA(qc,'pos')
    pos_RA = ForwKinRA(qc,'pos')
    rot_LA = ForwKinLA(qc,'rot')
    rot_RA = ForwKinRA(qc,'rot')

    pos_des_LA = SX([BoxPos[0],BoxPos[1] + L/2,BoxPos[2]])
    pos_des_RA = SX([BoxPos[0],BoxPos[1] - L/2,BoxPos[2]])
#    pos_des_LA = SX([1.0,0.3,1.1])
#    pos_des_RA = SX([1.1,0.2,0.8])
    rot_ref = np.matrix("0 0 -1; 0 1 0 ; 1 0 0")

    #Cost function due to position
    des = 1000*dot(pos_LA - pos_des_LA, pos_LA - pos_des_LA)
    des += 1000*dot(pos_RA - pos_des_RA, pos_RA - pos_des_RA)
    #Cost function due to angles
    des += 10*dot(rot_LA - rot_ref, rot_LA - rot_ref)
    des += 10*dot(rot_RA - rot_ref, rot_RA - rot_ref)

   
    w = vertcat(*w)
    
    #Nlp problem
    prob = dict(x = w, f = des)
    #Create solver interface
    opts = {"ipopt": {"print_level" : 0 }}
    solver = nlpsol('solver','ipopt', prob,opts)
    sol = solver(lbx = lbw , ubx = ubw)

    #
    qc_0 = sol['x']
    print(" ")
    print("##################################")
    print("# Centauro initial condition is: #")
    print("##################################")
 
    for l in range (0,mpc.nq):
        print("q" + str(l) + ": [" + str(qc_0[l]) + "]")
        

    print(" ")
    #Visualize initial condition
    InvTalker(qc_0)
    print("")
    return qc_0


def CheckInitialCondition(qc_0):
    for i in range(nq):
        if (Centauro_features.joint_angle_lb[i] > qc_0[i]) or (Centauro_features.joint_angle_ub[i] < qc_0[i] ):
            raise Exception('Invalid initial condition. Bounds exceeded')


        if i == 0: 
            print('Left arm  joint constraints:')
            print("")
        
        if i == 6: 
            print("")
            print("Right arm joint constraints:")
            print("")

        print(str(Centauro_features.joint_angle_lb[i]) + ' < ' + str(qc_0[i]) + ' < ' +  str(Centauro_features.joint_angle_ub[i]))

def CheckInitialVelocity(qcdot_0):
    for i in range(nq):
        if (- Centauro_features.joint_velocity_lim[i] > qcdot_0[i]) or (Centauro_features.joint_velocity_lim[i] < qcdot_0[i] ):
            raise Exception('Invalid initial velocity condition. Bounds exceeded')


        if i == 0: 
            print('Left arm  joint velocity constraints:')
            print("")
        
        if i == 6: 
            print("")
            print("Right arm joint velocity constraints:")
            print("")

        print(str( - Centauro_features.joint_velocity_lim[i]) + ' < ' + str(qcdot_0[i]) + ' < ' +  str(Centauro_features.joint_velocity_lim[i]))

def RelativePosition(qc_0):
    #E1 effector
    pos1 = ForwKinLA(qc_0,'pos')
    rot1 = ForwKinLA(qc_0,'rot')
    #E2 effector    
    pos2 = ForwKinRA(qc_0,'pos')
    rot2 = ForwKinRA(qc_0,'rot')
    
    H1 = vertcat( np.array(horzcat(rot1,pos1)),np.array([0.0,0.0,0.0,1]).reshape(1,4) )
    H2 = vertcat( np.array(horzcat(rot2,pos2)),np.array([0.0,0.0,0.0,1]).reshape(1,4) )
    
    
    pos1 = np.array(pos1).reshape(3,1)
    rot1 = np.array(rot1).reshape(3,3)
    
    pos2 = np.array(pos2).reshape(3,1)
    pos2 = np.append(pos2,1.0).reshape(4,1)
    
    
    mat_inv = np.vstack([np.hstack([rot1.T,-mtimes(rot1.T,pos1)]),
                        np.array([0,0,0,1]).reshape(1,4)])
    
    pRR = ForwKinRA(qc_0,'pos')
    pLL = ForwKinLA(qc_0,'pos')                      
    pos_2in1_w1 = mtimes(rot1.T,pRR) - mtimes(rot1.T,pLL)
    pos_2in1_w2 = mtimes(mat_inv,pos2)
    #pos_1in2_w2 = np.round(mtimes(mat_inv,pos2),1)
    return(pos_2in1_w1)


def RelativeOrientationError(qc):
    rot1 = ForwKinLA(qc,'rot')
    rot2 = ForwKinRA(qc,'rot')

    R_o = mtimes(rot1,rot2.T)
    #Compute the skew metrix of R_o
    R_skew = (R_o - R_o.T)/2

    #ex = np.round(R_skew[2,1],3)[0][0]
    #ey = np.round(R_skew[2,0],3)[0][0]
    #ez = np.round(R_skew[1,0],3)[0][0]
    #e_des = np.round(np.array([ex,ey,ez]).reshape(3,1),1)
    e_des = np.array([R_skew[2,1],R_skew[2,0],R_skew[1,0]])
    
    return(e_des)


    
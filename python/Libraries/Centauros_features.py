import mpc_fatigue.pynocchio_casadi as pin
from casadi import *
import numpy as np
import rospy
import csv

#ROS parameters
centauro = rospy.get_param('/robot_description')
end_LA = 'mass1_ee'
end_RA = 'mass2_ee'
# General parameters
nq = 14

    
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
def SaveCsv(folder,qc_opt,qcd_opt,F_opt,tau_LR,tau_RR,lbt,ubt,box_pos,box_rpy,Tw,lbtemp,ubtemp):

    #Save the optimal joint angles
    with open('/home/user/workspace/src/mpc_fatigue/Centauro_solutions/' + folder + '/qc_optimal.csv', 'wb') as myfile:
        wr = csv.writer(myfile)
        wr.writerow(qc_opt)
    
    #Save the optimal joint velocities
    with open('/home/user/workspace/src/mpc_fatigue/Centauro_solutions/' + folder + '/qcdot_optimal.csv', 'wb') as myfile:
        wr = csv.writer(myfile)
        wr.writerow(qcd_opt)
    
    #Save the optimal force at the two end effector
    with open('/home/user/workspace/src/mpc_fatigue/Centauro_solutions/' + folder + '/F_optimal.csv', 'wb') as myfile:
        wr = csv.writer(myfile)
        wr.writerow(F_opt)
    
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


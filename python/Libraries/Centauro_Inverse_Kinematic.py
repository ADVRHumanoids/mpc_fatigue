from casadi import *
from Centauros_features import *
import rospy
import numpy as np
import Centauros_inverse_kinematics_talker as ti



# General parameters
nq = 14


#From the desider initial condition, compute the inverse kinematic to determine
def InvKin(BoxPos,L):
    qc = SX.sym('qc', nq)
    g = []
    lbg = []
    ubg = []

    pos_LA = ForwKinLA(qc,'pos')
    pos_RA = ForwKinRA(qc,'pos')
    rot_LA = ForwKinLA(qc,'rot')
    rot_RA = ForwKinRA(qc,'rot')

    pos_des_LA = SX([BoxPos[0],BoxPos[1] + L/2,BoxPos[2]])
    print(pos_des_LA)
    pos_des_RA = SX([BoxPos[0],BoxPos[1] - L/2,BoxPos[2]])
    rot_ref = np.matrix("0 0 -1; 0 1 0 ; 1 0 0")

    #Cost function due to position
    des = 1000*dot(pos_LA - pos_des_LA, pos_LA - pos_des_LA)
    des += 1000*dot(pos_RA - pos_des_RA, pos_RA - pos_des_RA)
    #Cost function due to angles
    des += 10*dot(rot_LA - rot_ref, rot_LA - rot_ref)
    des += 10*dot(rot_RA - rot_ref, rot_RA - rot_ref)

    g.append(qc)
    lbg = vertcat(Centauro_features.joint_angle_lb)
    ubg = vertcat(Centauro_features.joint_angle_ub)
    g = vertcat(*g)
    #Nlp problem
    prob = dict(x = qc, f = des , g = g )
    #Create solver interface
    solver = nlpsol('solver','ipopt', prob)
    sol = solver(x0 = np.zeros(nq).tolist(),lbg = lbg , ubg = ubg)

    #
    qc_0 = sol['x']
    print(" ")
    print("Centauro initial condition is: ")
    print(str(qc_0))
    print(" ")
    #Visualize initial condition
    ti.talker(qc_0)
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


def InitialRelativePosition(qc_0):
    #E1 effector
    pos1= ForwKinLA(qc_0,'pos')
    rot1= ForwKinLA(qc_0,'rot')
    pos1 = np.array(pos1).reshape(3,1)
    pos1 = np.append(pos1,1.0).reshape(4,1)

    #E2 effector    
    pos2 = ForwKinRA(qc_0,'pos')
    rot2 = ForwKinRA(qc_0,'rot')

    pos2 = np.array(pos2).reshape(3,1)
    rot2 = np.array(rot2).reshape(3,3)


    mat_inv = np.vstack([np.hstack([rot2.T,-mtimes(rot2.T,pos2)]),
                        np.array([0,0,0,1]).reshape(1,4)])
                        
                        
    pos_1in2 = mtimes(mat_inv,pos1)
    return(pos_1in2)


def InitialRelativeOrientationError(qc_0):
    rot1= ForwKinLA(qc_0,'rot')
    rot2 = ForwKinRA(qc_0,'rot')

    R_o = mtimes(rot1,rot2.T)
    #Compute the skew metrix of R_o
    R_skew = (R_o - R_o.T)/2

    ex = np.round(R_skew[2,1],3)[0][0]
    ey = np.round(R_skew[2,0],3)[0][0]
    ez = np.round(R_skew[1,0],3)[0][0]

    e_des = np.array([ex,ey,ez]).reshape(3,1)
    return(e_des)
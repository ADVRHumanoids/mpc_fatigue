"""
Created by Pasquale Buonocore 30/10/2019 @IIT


In this script the inverse kinematics is tested.
Remember to run the Rviz environment

roslaunch mpc_fatigue 2_pilz_rviz_6DOF.launch
"""


###############################################################################
# --------------------------- Load libraries ---------------------------------#
###############################################################################

from casadi import *
import rospy
import mpc_fatigue.pynocchio_casadi as pin
import matplotlib.pyplot as plt
import numpy as np
import csv
import talker_inv_dyn_two_pilz_6DOF as ta
import two_pilz_talker_inv_kin_6DOF as ti

###############################################################################
#--------------------------What do you want to do? ---------------------------#
###############################################################################

#SET PLOT == True if you want to plot the solutions
#SET OCP == True if you want to solve the OCP again

class Check:
   OCP = True
   effectors_constraint = False
   Plot = True

###############################################################################
# ------------------------------ Functions -----------------------------------#
###############################################################################

if Check.Plot == True or Check.OCP == True:
    
#----------------------------  Load the urdf  --------------------------------#
    pilz_first = rospy.get_param('/robot_description_1')
    pilz_second = rospy.get_param('/robot_description_2')
    
    
# ---------------------- Solve forward kinematics pilz 1  --------------------#
    end_point = 'prbt_link_5' #prbt_link_5 #end_effector
    #Postion of link 5 in cartesian space (Solution to direct kinematics)
    fk_first_string = pin.generate_forward_kin(pilz_first, end_point)
    #Create casadi function
    fk_first = casadi.Function.deserialize(fk_first_string)
# ---------------------- Solve forward kinematics pilz 2  --------------------#
    #Postion of link 5 in cartesian space (Solution to direct kinematics)
    fk_second_string = pin.generate_forward_kin(pilz_second, end_point)
    #Create casadi function
    fk_second = casadi.Function.deserialize(fk_second_string)
    
# ---------------------- Solve Jacobian of link_5  ---------------------------#
    #Jacobian for link_5 (Solution to direct kinematics)
    jac_first_string = pin.generate_jacobian(pilz_first, end_point)
    jac_second_string = pin.generate_jacobian(pilz_second, end_point)
    #Create casadi function
    jac_first = casadi.Function.deserialize(jac_first_string)
    jac_second = casadi.Function.deserialize(jac_second_string)
# ---------------------- Solve Inverse dynamics Pilz 1 and 2  ----------------#
    Idyn_first_string = pin.generate_inv_dyn(pilz_first)
    Idyn_second_string = pin.generate_inv_dyn(pilz_second)
    #Create casadi function
    Idyn_first = casadi.Function.deserialize(Idyn_first_string)
    Idyn_second = casadi.Function.deserialize(Idyn_second_string)


###############################################################################
# --------------------------- Initial conditions -----------------------------#
###############################################################################

    # Define time 
    T = 2.
    N = 80
    h = T/N
    # Define a certain time grid
    tgrid = [T/N*k for k in range(N)]

    #Variables
    nq = 6
    # First arm variables
    qc_f = SX.sym('qc_f', nq) #joint angles
    # Second arm variables
    qc_s = SX.sym('qc_s', nq) #joint angles
    
    #Same zero acceleration and velocity for both
    qc_dot = np.zeros(nq) # Joint acceleration
    qcddot = np.zeros(nq) # Joint acceleration
    
    #Initial position end effector 1
    x_ini_first = 0.3
    y_ini_first = 0.2
    z_ini_first = 0.0
    p_init_first = np.array([x_ini_first,y_ini_first,z_ini_first])
    
    #Initial posizione end effector 2
    x_ini_second = 0.7
    y_ini_second = y_ini_first
    z_ini_second = z_ini_first
    p_init_second = np.array([x_ini_second,y_ini_second,z_ini_second])
    
    #Initial posizion box
    x_ini_box = (x_ini_first + x_ini_second)/2
    y_ini_box = y_ini_first
    z_ini_box = z_ini_first
    
    # Set the box final position
    x_box_des = 0.5
    y_box_des = - 0.3
    z_box_des = 0.4
    
    R1_ref = np.matrix('0 -1 0 ;-1 0 0 ;0 0 -1')
    R2_ref = np.matrix('0 0 -1 ;-1 0 0 ;0 1 0')
###############################################################################
# ------------------------ Solve forward kinematics --------------------------#
###############################################################################
    print('### Inverse kinematic solution ###')
# ----------------------- Solve forward kinematics pilz_1 --------------------#
    #From the desider initial condition, compute the inverse kinematic to determine
    pos_link_5_first = fk_first(q = qc_f)['ee_pos']
    pos_des_link_5 = SX([x_ini_first,y_ini_first,z_ini_first])

    R1_link_5 = fk_first(q = qc_f)['ee_rot']
    des = 100*dot(pos_link_5_first - pos_des_link_5, pos_link_5_first - pos_des_link_5)
    des += 10*dot(R1_link_5 - R1_ref, R1_link_5 - R1_ref)
    #Nlp problem
    prob_first = dict(x = qc_f, f = des )
    #Create solver interface
    solver_first = nlpsol('solver','ipopt', prob_first)
    sol_first = solver_first(x0 = [0.0,0.0,0.0,0.0,0.0,0.0])
    
# --------------------- Solve forward kinematics pilz_2 ----------------------#
    des = []
    #From the desider initial condition, compute the inverse kinematic to determine
    pos_link_5_second = fk_second(q = qc_s)['ee_pos']
    pos_des_link_5 = SX([x_ini_second,y_ini_second,z_ini_second])
    
    R2_link_5 = fk_second(q = qc_s)['ee_rot']
    des = 100*dot(pos_link_5_second - pos_des_link_5, pos_link_5_second - pos_des_link_5)
    des += 10*dot(R2_link_5 - R2_ref, R2_link_5 - R2_ref)
    #Nlp problem
    prob_second = dict(x = qc_s, f = des)
    #Create solver interface
    solver_second = nlpsol('solver','ipopt', prob_second)
    sol_second = solver_second(x0 = [0.0,0.0,0.0,0.0,0.0,0.0])
    
    #Initial position pilz 1 and pilz 2
    qc_f_init = sol_first['x']
    qc_s_init = sol_second['x']
    #Visualize initial condition
    ti.talker(qc_f_init,qc_s_init)

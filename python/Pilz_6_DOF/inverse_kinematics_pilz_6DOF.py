# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 14:40:45 2019
# ------ EXAMPLES ------#

#Solve inverse dynamics
inv_dyn_string = pin.generate_inv_dyn(urdf)

#DEfine states
qc = SX.sym('q', 6)

#Postion of link 5 in cartesian space (Solution to direct kinematics)
p = fk(q=qc)['ee_pos']

p_des = SX.zeros(3)

#Define the cost function
J = dot(p_des - p, p_des - p)

#Then define the solver e solve the inverse kinematics


@author: user
"""


#PURPOSE --> SOLVE INVERSE KINEMATIC

# --------------------------- Load libraries ---------------------------------
from casadi import *
import rospy
import mpc_fatigue.pynocchio_casadi as pin
import talker_inv_kin_pilz_3DOF as ti

#----------------------------  Load the urdf  ---------------------------------
urdf = rospy.get_param('/robot_description')

# ---------------------- Solve forward kinematics  ----------------------------
#Postion of link 5 in cartesian space (Solution to direct kinematics)
fk_string = pin.generate_forward_kin(urdf, 'prbt_link_5')
#Create casadi function
fk = casadi.Function.deserialize(fk_string)
#Look inside the fk function
print(fk)

#Variable q
qc = SX.sym('qc', 3)

pos_link_5 = fk(q = qc)['ee_pos']
pos_des_link_5 = SX([10.0,0.1,0.3])


J = dot(pos_link_5 - pos_des_link_5, pos_link_5 - pos_des_link_5)

#Nlp problem
prob = dict(x = qc, f = J)
#Create solver interface
solver = nlpsol('solver','ipopt', prob)

sol = solver(x0 = [0.0,0.0,0.0])
print(sol['x'])

q = sol['x']

ti.talker(q)






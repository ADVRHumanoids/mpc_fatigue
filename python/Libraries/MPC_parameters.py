#!/usr/bin/env python
# license removed for brevity

#####################################################
#                   PARAMETERS                      #
#####################################################

# Here you can change all the parameters of the MPC.
# The mpc principal node and unroller node take the parameters from here
import numpy as np

class mpc:
    #Here you can set the frequency at which the OCP solver will will send the obtained optimal trajectory and force to the unroller node
    OCP_solver_publish_rate = 1000
    unroller_publish_rate = 120
    initial_position_rate = 1000
    
    #Here you can set the simulation time of each iteration and the number of the DMS method
    T = 20.  #20
    N = 40   #30
    Tperc = 0.99
    #Cost function weight
    boxdist_w = 100
    joint_vel_w = 10
    force_w = 0.0
    
    #According the way you set those parameters, you can change constraints and solver
    Const1 = True               # Const1 = True will use the FULL POSITION CONSTRAINTS 
    Const2 = False              # Const2 = True will use the FULL VELOCITY CONSTRAINTS 
    Const3 = False              # Const3 = True will use the HYBRID CONSTRAINTS: 3 POSITIONS + 3 ANGULAR VELOCITIES 
    Const4 = False              # Const4 = True will use the HYBRID CONSTRAINTS: 3 LINEAR VELOCITIES + 3 ORIENTATION ANGLES 
    MySolver = "ipopt"          # sqpmethod or ipopt'''

    #Here you can define the initial position of the box and its length
    box_initial_position = np.array([0.9, 0.0 , 1.0]) #0.9 0.0 1.0
    Lbox = 0.4 #0.4 

    
    #System degree of freedom
    nq = 14
    
    #Mass
    m = 10  #10  
    
    #Constraint tollerance
    constraint_tollerance = 0.001
    
    #Temperatue bound
    temperature_bound = 80.0  #80.0
    
    #Friction coefficient
    mu = 0.3   #0.3
    
    
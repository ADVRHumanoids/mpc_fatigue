# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 14:20:52 2019

@author: user
"""

"""
Created by Pasquale Buonocore 30/10/2019 @IIT


In this script the inverse kinematics is tested.
Remember to run the Rviz environment

roslaunch mpc_fatigue 2_pilz_rviz_6DOF.launch
"""


###############################################################################
# --------------------------- Load libraries ---------------------------------#
###############################################################################
import mpc_fatigue.pynocchio_casadi as pin
import Centauros_inverse_kinematics_talker as ti
import Centauro_dynamics_talker as ta
import TemperatureModel as TempModel
from Centauro_Inverse_Kinematic import *
from Centauros_features import *
from casadi import *
import numpy as np
import rospy
import csv
import time
import matplotlib.pyplot as plt
###############################################################################
#--------------------------What do you want to do? ---------------------------#
###############################################################################


class Check:
   OCP = True
   Rviz = True
   
   Const1 = True
   Const2 = False
   
   Const3 = False
   Const4 = False
   
   '''
   Const1 = True will use the FULL POSITION CONSTRAINTS
   
   Const2 = True will use the FULL VELOCITY CONSTRAINTS
   
   Const3 = True will use the HYBRID CONSTRAINTS: 3 POSITIONS + 3 ANGULAR VELOCITIES
   
   Const4 = True will use the HYBRID CONSTRAINTS: 3 LINEAR VELOCITIES + 3 ORIENTATION ANGLES
   '''

# Default folder where all the csv file are saved
folder = "CsvSolution/"

# Define string constraint 

Sl = [1,1,1,0,0,0,0]
Cl = [0.,0.,0.,0.,0.,0.,0.]

Sr = [0,0,0,0,0,0,0]
Cr = [0.,0.,0.,0.,0.,0.,0.]

S = Sl + Sr
C = Cl + Cr

ktau = 40

###############################################################################
#                               Initialization                                #
###############################################################################

print('### Inverse kinematic solution ###')
print("")

#Set box initial positions
Box_ini = np.array([0.9, 0.0 , 1.3])
#Set box length
Lbox = 0.4
#Solve inverse kinematics and plot the solution in Rviz
qc_0 = InvKin(Box_ini,Lbox)
Rdes = ForwKinLA(qc_0,'rot')
#Check initial condition
print("####### Checking initial condition ###### ")
print("")
CheckInitialCondition(qc_0)


###############################################################################
# -------------------------- START OCP PROBLEM -------------------------------#
###############################################################################

# Define time 
T = 2.
N = 50
h = T/N


# Define a certain time grid
tgrid = [T/N*k for k in range(N)]

#System degree of freedom
nq = 14

# Mass to lift up
m = 10 # [ Kg ]

#Tollerance
toll = 0.001 # [N] on [Nm]

#Same zero acceleration and velocity for both arms at step 0
qc_dot0 = np.zeros(nq).tolist() # Joint acceleration
qc_ddot0 = np.zeros(nq).tolist() # Joint acceleration

# Joint angles bound
lj = Centauro_features.joint_angle_lb # [rad]
uj = Centauro_features.joint_angle_ub # [rad]

# Joint velocity bound
lbqdot =  - Centauro_features.joint_velocity_lim # [rad/s]
ubqdot = Centauro_features.joint_velocity_lim # [rad/s]

#Torque bounds
lbtorque = -Centauro_features.joint_torque_lim # [N/m]
ubtorque = Centauro_features.joint_torque_lim # [N/m]

#COMPUTE INITIAL RELATIVE POSITION AND RELATIVE ERROR
#Left End effector wrt right end effector
RelativePosition_0 = InitialRelativePosition(qc_0)
RelativeOrientation_0 = InitialRelativeOrientationError(qc_0)

#Initialilize the relative position list
RelPosition = [RelativePosition_0]


#Compute the initial torques
WA0 = np.array([0,0,m*9.81/2,0,0,0]).reshape(6,1)
tau0 = InvDyn(qc_0,qc_dot0,qc_ddot0) + vertcat(mtimes(Jac_LA(qc_0).T,WA0),mtimes(Jac_RA(qc_0).T,WA0))
#From the initial torques get the current
Ic = np.round(tau0/ktau,4)

T_0 = np.full((1,nq),20.0)[0].tolist() 
                
if Check.OCP == True:
    print('')
    print('############## Optimal control problem ############## ')
    
    #   EMPTY NLP 
    w = []
    lbw = []
    ubw = []
    
    #Constraint lists
    g = []
    lbg = []
    ubg = []
    
    #Torque constraints to plot lists
    lbt = []
    ubt = []

    #Cost function
    J = 0

    #Box weight
    Fdes = SX(9.81 * m)

    # nq states each arm
    qc_k = SX.sym('qc0', nq)
    w.append(qc_k)
    lbw +=  [qc_0] 
    ubw +=  [qc_0] 
    
    for k in range(N):
        
        R01 = ForwKinLA(qc_k,'rot')
        R02 = ForwKinRA(qc_k,'rot')
        pL = ForwKinLA(qc_k,'pos')
        pR = ForwKinRA(qc_k,'pos')
        JLA = Jac_LA(qc_k)
        JRA = Jac_RA(qc_k)
        
        ###################################################   
        #                    CONTROLS                     #   
        ###################################################
        
        #Control at each interval
        qcdname = 'qcd'+ str(k)
        qcd_k = SX.sym(qcdname, nq)
        w.append(qcd_k)
        if k == 0:
            lbw +=  qc_dot0
            ubw +=  qc_dot0
        else:
            lbw +=  [ -Centauro_features.joint_velocity_lim ]
            ubw +=  [ Centauro_features.joint_velocity_lim ]
    
    
        ###################################################   
        #    NEW POSITION, FORCE AND APPLICATION POINT    #   
        ###################################################

        # Gravitational force application point
        pbox = (pL + pR)/2
                 
        #Force that the left end effector exerts to the object
        moment_component = SX.zeros(3)
        F_name = 'FLR'+ str(k)
        F_LR = SX.sym(F_name,3) 
        w.append(F_LR)
        lbw +=  np.full((1,F_LR.shape[0]),-np.inf)[0].tolist() 
        ubw +=  np.full((1,F_LR.shape[0]),np.inf)[0].tolist() 
        W_LA = vertcat(F_LR,moment_component)
        
        #Force that the right end effector exerts to the object
        F_name = 'FRR'+ str(k)
        F_RR = SX.sym(F_name,3) 
        w.append(F_RR)
        lbw +=  np.full((1,F_RR.shape[0]),-np.inf)[0].tolist() 
        ubw +=  np.full((1,F_RR.shape[0]),np.inf)[0].tolist() 
        W_RA = vertcat(F_RR,moment_component)
    
        ###################################################  
        ##                 CONSTRAINTS                   ##   
        ###################################################
        
        #Force static equilibrium
        g.append(vertcat( F_LR[2] + F_RR[2] - Fdes, F_LR[0] + F_RR[0], F_LR[1] + F_RR[1]))
        lbg +=  np.full((1,3),-toll)[0].tolist() 
        ubg +=  np.full((1,3),toll)[0].tolist() 
        
        #Moment static equilibrium
        g.append(vertcat(cross(pL - pR,F_LR) + cross(pR - pL,F_RR)))
        lbg +=  np.full((1,3),-toll)[0].tolist() 
        ubg +=  np.full((1,3),toll)[0].tolist() 
    
        # RELATIVE POSE CONSTRAINT
        if Check.Const1 == True or Check.Const3 == True or Check.Const4 == True : 
            
            #pos1 = vertcat(pos1,SX(1.0))
            
            if Check.Const1 == True or Check.Const3 == True:
                #### RELATIVE POSITION ####
#                    mat_inv = vertcat(horzcat(rot2.T,-mtimes(rot2.T,pos2)),SX([0.0,0.0,0.0,1.0]).T)
#                    #Compute the relative position of the E1 in E2
#                    pos_1in2 = mtimes(mat_inv,pos1)
                pos_1in2 = mtimes(R01.T,pR) - mtimes(R01.T,pL)
                
                #Add the constraint - The relative position should be the same
                g.append(pos_1in2[0:3] - RelPosition[k][0:3]) 
                #g.append(dot(pos_1in2[0:3] - RelPosition[k][0:3],pos_1in2[0:3] - RelPosition[k][0:3]))
#                    lbg += np.zeros(3).tolist()  
#                    ubg += np.zeros(3).tolist()
                lbg +=  np.full((1,3),0.0)[0].tolist() 
                ubg +=  np.full((1,3),0.0001)[0].tolist() 
                RelPosition.append(pos_1in2)
            
            if Check.Const1 == True or Check.Const4 == True:
                #### RELATIVE ORIENTATION ####
                R_o = mtimes(R01,R02.T)
                #Compute the skew metrix of R_o
                R_skew = (R_o - R_o.T)/2
                #Extract errors
                ex = R_skew[2,1]
                ey = R_skew[2,0]
                ez = R_skew[1,0]
                e_k = np.array([ex,ey,ez])
                
                g.append(reshape(e_k,(3,1)) - RelativeOrientation_0)
                lbg += np.zeros(3).tolist()  
                ubg += np.zeros(3).tolist()
#                    lbg +=  np.full((1,3),0.0)[0].tolist() 
#                    ubg +=  np.full((1,3),0.0001)[0].tolist()  

        # RELATIVE TWIST CONSTRAINT
        if Check.Const2 == True or Check.Const3 == True or Check.Const4 == True:
            
            #Compute pos 2 in 1
            p = mtimes(R01.T,pR) - mtimes(R01.T,pL)
            #From pos 2 in 1 I can compute the skew matrix
            I = np.array([1,0,0,0,1,0,0,0,1]).reshape(3,3)
            zerome = np.zeros(9).reshape(3,3)
            
            Sp2in1 = np.array([0, -p[2], p[1],p[2],0,-p[0],-p[1], p[0],0]).reshape(3,3)
            
            #IMPORTANT PSI MATRIX
            psi = vertcat(horzcat(I, - Sp2in1), horzcat(zerome, I))
            
            #IMPORTANT ROTATION MATRIX
            omega = vertcat(horzcat(R01.T, zerome), horzcat(zerome, R01.T))
            
            partA = mtimes(mtimes(-psi,omega),JLA)
            partB = mtimes(omega,JRA)
            
            
            Jr = horzcat(partA , partB)
            
            if Check.Const2 == True:
                g.append(mtimes(Jr,qcd_k))
                lbg += np.zeros(6).tolist()  
                ubg += np.zeros(6).tolist()
            elif Check.Const3 == True: # Angular velocities
                g.append(mtimes(Jr,qcd_k)[3:6])
                lbg += np.zeros(3).tolist()  
                ubg += np.zeros(3).tolist()
            elif Check.Const4 == True: #Linear velocities
                g.append(mtimes(Jr,qcd_k)[0:3])
                lbg += np.zeros(3).tolist()  
                ubg += np.zeros(3).tolist()
                

        #I have defined the force that the robot exerts on the ambient.
        #The force that the ambient exerts on the robot is equal with a minus sign
        #You need to take care of that while computing the inverse dynamics

        #Compute torque at each step
        JLA = Centauro_features.jac_la(qc_k)
        JRA = Centauro_features.jac_ra(qc_k)
        tau = InvDyn(qc_k, qcd_k, qc_ddot0) + mtimes(JLA.T,W_LA) + mtimes(JRA.T,W_RA)
        
        #Based on the S string, parse it and constraint the wanted motors
        for i in range(np.size(S)):
            
            if S[i] == 1:
                g.append(tau[i])
                if k < int(N/3):
                    lbg += [lbtorque[i]]
                    ubg += [ubtorque[i]]
                    lbt.append(lbtorque[i])
                    ubt.append(ubtorque[i])
                else:
                    lbg += [ - C[i]]
                    ubg += [C[i]]
                    lbt.append(- C[i])
                    ubt.append(C[i])
            else:
                g.append(tau[i])
                lbg += [lbtorque[i]]
                ubg += [ubtorque[i]]
                lbt.append(lbtorque[i])
                ubt.append(ubtorque[i])
    
        ###################################################   
        ##                 COST FUNCTION                 ##   
        ###################################################
        J += 100*dot(pbox - Box_ini , pbox - Box_ini)
        J += mtimes(qcd_k.T,qcd_k)
        ###################################################   
        ##                 NEW VARIABLES                 ##   
        ###################################################
        
        #Integration
        q_next = qc_k + qcd_k * h
        #Integrate temperature
        #Tw.append(  np.e**(-T/N/Ttheta) * Tw[i] + Ploss * Rtheta * (1 - np.e**(-T/N/Ttheta)) )
        
        #New local state
        qname = 'qc' + str(k+1)
        qc_k= SX.sym(qname,nq)
        w.append(qc_k)
        lbw += lj.tolist()
        ubw += uj.tolist()
            
        #Continuity constraint
        g.append(q_next - qc_k)
        lbg +=  np.zeros(nq).tolist() 
        ubg +=  np.zeros(nq).tolist()

    #Vertcat all the symbolic lists    
    ubg = vertcat(*ubg)
    lbg = vertcat(*lbg)
    ubw = vertcat(*ubw)
    lbw = vertcat(*lbw)
    g = vertcat(*g)
    w = vertcat(*w)
    
    #Check if the lists have the correct dimensions
    print("g.shape:",g.shape)
    print("lbg.shape:",lbg.shape)
    print("ubg.shape:",ubg.shape)
    print("x.shape:",w.shape)
    print("lbw.shape:",lbw.shape)
    print("ubw.shape:",ubw.shape)
    
    # Create the nlp solver
    nlp = dict(f = J, g = g, x = w)

    Solver = nlpsol('Solver','ipopt',nlp)
    r = Solver(lbx = lbw, ubx = ubw, lbg = lbg, ubg = ubg)   
        
    sol = r['x'].full().flatten()
    
    
    
    with open('/home/user/workspace/src/mpc_fatigue/Centauro_solutions/' + folder + '/solution.csv', 'wb') as myfile:
        wr = csv.writer(myfile)
        wr.writerow(sol)
            
            
            
                ###################################################   
                #       EXTRACT VARIABLES AND SAVE CSV FILE       #   
                ###################################################
    if Check.OCP == True or Check.Rviz == True:
        
        print('')
        print('############## Creating csv file ############## ')
        
        sol = []
        
        with open('/home/user/workspace/src/mpc_fatigue/Centauro_solutions/' + folder + '/solution.csv', 'rb') as solution:
            for line in solution:
                x = line.split(',')
                for val in x:
                    sol.append(float(val))

           
        nf = 3 # force component
        nq = 14
        n = 2*nq + 2*nf # element solution of each step        
    
        #Empty lists
        qc_opt = []
        qcd_opt = []
        F_opt = []
        tau_LR = []
        tau_RR = []
    
        #Variables from the solution
        k = 0
        while (k < N):
            qc_opt.append(sol[k*n:k*n+nq])
            qcd_opt.append(sol[k*n+nq:k*n+2*nq])
            F_opt.append(sol[k*n+2*nq:k*n+n])
            J_LA = Jac_LA(qc_opt[k])
            J_RA = Jac_RA(qc_opt[k])
            W_LA = vertcat(sol[k*n+2*nq:k*n+n][0:nf],np.zeros(3).tolist())
            W_RA = vertcat(sol[k*n+2*nq:k*n+n][nf:2*nf],np.zeros(3).tolist())
            Inv_dyn = InvDyn(qc_opt[k], qcd_opt[k], qc_ddot0) + vertcat(mtimes(J_LA.T,W_LA), mtimes(J_RA.T,W_RA))
            tau_LR.append(Inv_dyn[0:7])
            tau_RR.append(Inv_dyn[7:14])
            k+= 1
        
        #Box position and orientation
        k = 0
        box_pos = []
        box_rpy = []
        while k < N :
            E1 = ForwKinLA(sol[k*n:k*n+nq],'pos')
            E2 = ForwKinRA(sol[k*n:k*n+nq],'pos')
            R1 = ForwKinLA(sol[k*n:k*n+nq],'rot')
            
            Roll = np.arctan2(R1[2,0],R1[2,1])
            Pitch = np.arccos( -R1[2,2])
            Yaw = - np.arctan2(R1[0,2],R1[1,2])
            
            box_rpy.append([Roll,Pitch,Yaw])
            box_pos.append(np.round((E1 + E2)/2,3))
            k = k + 1
            
         
        #Make a unique list
        qc_opt = np.concatenate(qc_opt).tolist()
        qcd_opt = np.concatenate(qcd_opt).tolist()
        F_opt = np.concatenate(F_opt).tolist()
        tau_LR = np.concatenate(np.concatenate(tau_LR).tolist())
        tau_RR = np.concatenate(np.concatenate(tau_RR).tolist())
        box_pos = np.concatenate(np.concatenate(box_pos))
        box_rpy = np.concatenate(box_rpy)
    
        if Check.Rviz == True:
                    
            print('')
            print('############## Rviz simulation ############## ')
        
            ta.talker(qc_opt)  
            
        #Save the csv file
        SaveCsv(folder,qc_opt,qcd_opt,F_opt,tau_LR,tau_RR,lbt,ubt,box_pos,box_rpy)

print('Done')    

        

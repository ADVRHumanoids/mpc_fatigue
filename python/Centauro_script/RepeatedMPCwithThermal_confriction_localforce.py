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
import Centauro_dynamics_talker as ta
from Tmodel_library import *
from Centauro_functions import *
from casadi import *
import numpy as np
import rospy
import csv
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

# Pick an NLP solver
MySolver = "ipopt"
#MySolver = "worhp"
#MySolver = "sqpmethod"
#MySolver = "snopt"
#Solver = "ipopt_sqp"

rospy.init_node('node_position_from_invkin')
###############################################################################
#                               Initialization                                #
###############################################################################
print('### Inverse kinematic solution ###')
print("")

#Set box initial positions
Box_ini = mpc.box_initial_position
#Set box length
Lbox = mpc.Lbox
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
#T = 30.
#N = 40
T = mpc.T
N = mpc.N
h = T/N

# Define a certain time grid
tgrid = [T/N*k for k in range(N)]

#System degree of freedom
nq = mpc.nq

# Mass to lift up
m = mpc.m # [ Kg ]

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

lbtemp = np.full((1,nq),0.0)[0].tolist()
ubtemp = np.full((1,nq),mpc.temperature_bound)[0].tolist()

#COMPUTE INITIAL RELATIVE POSITION AND RELATIVE ERROR

#Left End effector wrt right end effector
#position of 2 wrt 1
RelativePosition_0 = InitialRelativePosition(qc_0)
RelativeOrientation_0 = InitialRelativeOrientationError(qc_0)

#Initialilize the relative position list
RelPosition = [RelativePosition_0]


#Compute the initial torques
WA0 = np.array([0,0,m*9.81/2,0,0,0]).reshape(6,1)

tau0 = InvDyn(qc_0,qc_dot0,qc_ddot0) + vertcat(mtimes(Jac_LA(qc_0).T,WA0),mtimes(Jac_RA(qc_0).T,WA0))
#From the initial torques get the current
T_0 = np.full((1,nq),20.0)[0].tolist() 

soltot = []
s = 0
iterations = 1

#Friction coefficient
mu = 0.4


#INITIAL CONDITION FOR THE FIRST WARMSTART
#qc_0 = np.array(qc_0).reshape(1,14)[0].tolist()
#F0 = np.array([0,0,m*9.81/2,0,0,m*9.81/2]).reshape(1,6)[0].tolist()
#sol0 = qc_0 + T_0 + qc_dot0 + F0
#sol0 = sol0 * N + qc_0 + T_0 

L = mpc.Lbox
BoxPos = mpc.box_initial_position

if Check.OCP == True:
    while s < iterations:    
        
        #   EMPTY NLP 
        w = []
        lbw = []
        ubw = []
        
        #Constraint lists
        g = []
        lbg = []
        ubg = []
        
        #Torque constraints to plot lists
        lbtq = []
        ubtq = []
        lbTe = []
        ubTe = []
    
        #Cost function
        J = 0
    
        #Box weight
        Fdes = SX([0,0, - 9.81 * m])
        
    
        print('')
        print('############## Optimal control problem ############## ')
        
        # nq states each arm
        qc_k = SX.sym('qc0', nq)
        w.append(qc_k)
        lbw +=  [qc_0] 
        ubw +=  [qc_0] 
        
        T_k = SX.sym('T0', nq)
        w.append(T_k)
        lbw +=  T_0
        ubw +=  T_0 
        lbTe.append(lbtemp)
        ubTe.append(ubtemp)
        
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
                     
            #Force that the left end effector exerts to the object in LOCAL FRAME that the enviorment excert on the robot
            moment_component = SX.zeros(3)
            F_name = 'FLR'+ str(k)
            F_LR = SX.sym(F_name,3) 
            w.append(F_LR)
            lbw +=  np.full((1,3),-np.inf)[0].tolist() 
            ubw +=  np.full((1,3),np.inf)[0].tolist() 
            W_LA = vertcat(F_LR,moment_component)
            
            #Force that the right end effector exerts to the object in LOCAL FRAME
            F_name = 'FRR'+ str(k)
            F_RR = SX.sym(F_name,3) 
            w.append(F_RR)
            lbw +=  np.full((1,3),-np.inf)[0].tolist() 
            ubw +=  np.full((1,3),np.inf)[0].tolist() 
            W_RA = vertcat(F_RR,moment_component)
        
            ###################################################  
            ##                 CONSTRAINTS                   ##   
            ###################################################
        
           
            #Friction cone contact 1
            A1 = np.matrix([ [0, -1,0],\
                             [0, - mu, 1], \
                             [0, - mu,-1], \
                             [1, - mu, 0], \
                             [-1, -mu, 0]])
            
            C1 = mtimes(A1,F_LR)

            #Friction cone contact 2
            A2 = np.matrix([ [0 , 1 , 0],\
                             [1 , mu , 0 ], \
                             [-1 , mu , 0], \
                             [0 , mu , 1 ], \
                             [0 , mu ,-1 ]])
                             
        
            C2 = mtimes(A2,F_RR)
            
            #Append the constraints
            g.append(vertcat(C1,C2))
            lbg += np.full((1,10),-np.inf)[0].tolist()  
            ubg += np.zeros(10).tolist()
            
            # Force static equilibrium
            # Lo voglio fare un globale e considerando le forze che il robot esercita sul box
            #Devo tornare in terna mondo e cambio di segno per avere le forze che il robot esercita sull'enviornment
            
            F_LR_mondo = mtimes(R01,F_LR)
            F_RR_mondo = mtimes(R02,F_RR)
            g.append(vertcat( - F_LR_mondo - F_RR_mondo + Fdes))
            lbg += np.zeros(3).tolist()  
            ubg += np.zeros(3).tolist()
            
            #Moment static equilibrium
            g.append(vertcat(cross(pL - pR,F_LR_mondo) + cross(pR - pL,F_RR_mondo)))
            lbg += np.zeros(3).tolist()  
            ubg += np.zeros(3).tolist()
            
            if Check.Const1 == True:
                # RELATIVE POSE CONSTRAINT
      
                #### RELATIVE POSITION ####
                pos_2in1 = mtimes(R01.T,pR) - mtimes(R01.T,pL)
                #Add the constraint - The relative position should be the same
                g.append(pos_2in1[0:3] - RelPosition[k][0:3])  
                lbg += np.zeros(3).tolist()  
                ubg += np.zeros(3).tolist()
                RelPosition.append(pos_2in1)
                    
                    
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
    
            # RELATIVE TWIST CONSTRAINT
            if Check.Const2 == True:
                
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
                
                    
            #I have defined the force that the robot exerts on the ambient.
            #The force that the ambient exerts on the robot is equal with a minus sign
            #You need to take care of that while computing the inverse dynamics

            #Compute torque at each step
            JLA = Centauro_features.jac_la(qc_k)
            JRA = Centauro_features.jac_ra(qc_k)
            #Con il segno più è il robot sull'ambiente
            #Con il segno meno è l'ambiente sul robot
            # SOno forze espresse nel sistema di riferimento globale
            tau = InvDyn(qc_k, qc_dot0, qc_ddot0) - mtimes(JLA.T,vertcat(F_LR_mondo,moment_component)) - mtimes(JRA.T,vertcat(F_RR_mondo,moment_component))
#            
            g.append(tau)
            lbg += [lbtorque]
            ubg += [ubtorque]
            lbtq.append(lbtorque)
            ubtq.append(ubtorque)
                        
            ###################################################   
            ##                 COST FUNCTION                 ##   
            ###################################################     
            J += 1000*dot(pbox - Box_ini , pbox - Box_ini)
            J += 100*dot(qcd_k,qcd_k)
#            J += 10*dot(F_LR,F_LR)
#            J += 10*dot(F_RR,F_RR)

            
            ###################################################   
            ##                 NEW VARIABLES                 ##   
            ###################################################
            
            #Integration
            q_next = qc_k + qcd_k * h

            #Integrate temperature
            Ia = tau/ktau
            Pj = (Ia**2) * Ra
            # Second component is due to all the other losses function of the motor speed
            Ps = (qcd_k**2) / Rh
            Ploss = Pj + Ps
            T_next = np.e**(-T/N/Ttheta) * T_k + Ploss * Rtheta * (1 - np.e**(-T/N/Ttheta))
            
            #New local state
            qname = 'qc' + str(k+1)
            qc_k= SX.sym(qname,nq)
            w.append(qc_k)
            lbw += lj.tolist()
            ubw += uj.tolist()
            
            Tname = 'T' + str(k+1)
            T_k = SX.sym(Tname, nq)
            w.append(T_k)
            lbw += lbtemp
            ubw += ubtemp
            lbTe.append(lbtemp)
            ubTe.append(ubtemp)
        
        
            #Continuity constraint
            g.append(q_next - qc_k)
            lbg +=  np.zeros(nq).tolist() 
            ubg +=  np.zeros(nq).tolist()
            
            #Continuity constraint
            g.append(T_next - T_k)
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
        #"tol": 10
        #'max_cpu_time' : 10
        #'acceptable_obj_change_tol': 10^22
        #'max_cpu_time' : 10
        #'hessian_approximation': 'limited-memory'
        #'jacobian_approximation' : 'finite-difference-values'
        #"linear_solver"
        #"acceptable_iter" : 2
        #opts = {"ipopt": {'tol' : 0.01,'acceptable_obj_change_tol': 10^10,"acceptable_iter" : 10}}
        #
        
        if s >= 1:
#            opts = {"ipopt": {"warm_start_init_point": "yes","tol": 0.001,"acceptable_tol": 0.001,"constr_viol_tol" : 0.001, "compl_inf_tol": 0.001}}
            opts = {"ipopt": {"tol": 0.001,"acceptable_tol": 0.001,"constr_viol_tol" : 0.001, "compl_inf_tol": 0.001}}

        else:
            opts = {"ipopt": {"tol": 0.001,"acceptable_tol": 0.001,"constr_viol_tol" : 0.001, "compl_inf_tol": 0.001}}
            #opts = {"ipopt": {"warm_start_init_point": "yes","tol": 0.001,"acceptable_tol": 0.001,"constr_viol_tol" : 0.001, "compl_inf_tol": 0.001}}

            
        #Allocate a solver
        #Solver = nlpsol("solver", MySolver, nlp, opts)
        Solver = nlpsol('Solver','ipopt',nlp)
        if s >= 1:
            r = Solver(x0 = sol,lbx = lbw, ubx = ubw, lbg = lbg, ubg = ubg)  
        else:
            #sol0 = np.array([qc_0,T_0,qc_dot0,[0,0,m*9.81/2,0,0,m*9.81/2]])
            r = Solver(lbx = lbw, ubx = ubw, lbg = lbg, ubg = ubg)  
            
#        Solver = nlpsol('Solver','sqpmethod',nlp)
#        r = Solver(lbx = lbw, ubx = ubw, lbg = lbg, ubg = ubg)  
        sol = r['x'].full().flatten()
        
        soltot.append(sol)
        s = s + 1
        
        #Here define the final condition that will be initial for the later approximation
        nf = 3 # force components
        nq = 14
        n = 3*nq + 2*nf # element solution of each step    
        qc_0 = sol[N*n:N*n+nq]
        T_0 = sol[N*n+nq:N*n+2*nq] - 0.05 
        qc_dot0 = sol[k*n+2*nq:k*n+3*nq]
        
        qc_0 = np.round(qc_0,4).tolist()
        T_0 = np.round(T_0,4).tolist()
        qc_dot0 = np.round(qc_dot0,4).tolist()
        
        RelativePosition_0 = np.round(InitialRelativePosition(qc_0),3)
        RelativeOrientation_0 = np.round(InitialRelativeOrientationError(qc_0),3)

        #Initialilize the relative position list
        RelPosition = [RelativePosition_0]

        print("Temperature: ")
        print(T_0)
        
        print("End effector relative position: ")
        print(RelativePosition_0)
        
        print("End effector relative orientation: ")
        print(RelativeOrientation_0)
        
        
        with open('/home/user/workspace/src/mpc_fatigue/Centauro_solutions/' + folder + '/solution.csv', 'wb') as myfile:
            wr = csv.writer(myfile)
            wr.writerow(sol)
            
    #Fine del while Salvo i dati
    soltot = np.concatenate(soltot)
        
    with open('/home/user/workspace/src/mpc_fatigue/Centauro_solutions/' + folder + '/sol_MPC.csv', 'wb') as myfile:
        wr = csv.writer(myfile)
        wr.writerow(soltot)

                ###################################################   
                #       EXTRACT VARIABLES AND SAVE CSV FILE       #   
                ###################################################
if Check.OCP == True or Check.Rviz == True:
    
    print('')
    print('############## Creating csv file ############## ')
    
    solTOT = []        
    with open('/home/user/workspace/src/mpc_fatigue/Centauro_solutions/' + folder + '/sol_MPC.csv', 'rb') as solution:
        for line in solution:
            x = line.split(',')
            for val in x:
                solTOT.append(float(val))

    sol_dim_cycle = np.size(solTOT)/iterations
    nf = 3 # force component
    nq = 14
    n = 3*nq + 2*nf # element solution of each step        

    #Empty lists
    qc_opt = []
    qcd_opt = []
    F_opt = []
    tau_LR = []
    tau_RR = []
    Tw = []
    box_pos = []
    box_rpy = []
    F_loc = []
    
    #Variables from the solution
    for i in range(iterations):
        sol = solTOT[sol_dim_cycle*i:sol_dim_cycle*i+sol_dim_cycle]
        k = 0
        j = 0
        print(i)
        while (k < N):
            qc_opt.append(sol[k*n:k*n+nq])
            Tw.append(sol[k*n+nq:k*n+2*nq])
            qcd_opt.append(sol[k*n+2*nq:k*n+3*nq])
            F_opt.append(sol[k*n+3*nq:k*n+n])
            R10 = np.array(ForwKinLA(sol[j*n:j*n+nq],"rot").T)
            R20 = np.array(ForwKinRA(sol[j*n:j*n+nq],"rot").T)
            F_loc.append(mtimes(R10,sol[k*n+3*nq:k*n+n][0:3]))
            F_loc.append(mtimes(R20,sol[k*n+3*nq:k*n+n][3:6]))
            J_LA = Jac_LA(qc_opt[k])
            J_RA = Jac_RA(qc_opt[k])
            W_LA = vertcat(sol[k*n+3*nq:k*n+n][0:nf],np.zeros(3).tolist())
            W_RA = vertcat(sol[k*n+3*nq:k*n+n][nf:2*nf],np.zeros(3).tolist())
            Inv_dyn = InvDyn(sol[k*n:k*n+nq], sol[k*n+2*nq:k*n+3*nq], qc_ddot0) + vertcat(mtimes(J_LA.T,W_LA), mtimes(J_RA.T,W_RA))
            tau_LR.append(Inv_dyn[0:7])
            tau_RR.append(Inv_dyn[7:14])
            k+= 1
        
            #Box position and orientation
            while j < N :
                E1 = ForwKinLA(sol[k*n:k*n+nq],'pos')
                E2 = ForwKinRA(sol[k*n:k*n+nq],'pos')
                R1 = ForwKinLA(sol[k*n:k*n+nq],'rot')
                
                Roll = np.arctan2(R1[2,0],R1[2,1])
                Pitch = np.arccos( -R1[2,2])
                Yaw = - np.arctan2(R1[0,2],R1[1,2])
                
                box_rpy.append([Roll,Pitch,Yaw])
                box_pos.append(np.round((E1 + E2)/2,3))
                j = j + 1
            
         
    #Make a unique list
    qc_opt = np.concatenate(qc_opt).tolist()
    qcd_opt = np.concatenate(qcd_opt).tolist()
    F_opt_glob = np.concatenate(F_opt).tolist()
    F_opt_loc = np.concatenate(np.concatenate(F_loc).tolist()).tolist()  
    tau_LR = np.concatenate(np.concatenate(tau_LR).tolist())
    tau_RR = np.concatenate(np.concatenate(tau_RR).tolist())
    box_pos = np.concatenate(np.concatenate(box_pos))
    box_rpy = np.concatenate(box_rpy)
    Tw = np.concatenate(Tw)
    frames = 30 * T
    new_qc_opt = []
    for j in range(nq):
        q_opt = qc_opt[j::14]
        q_opt_30fps = []
    
        for i in range(np.size(q_opt)-1):
            q_opt_30fps.append(np.linspace(q_opt[i],q_opt[i+1],frames/N+1,endpoint=False))
        
        q_opt_30fps = np.concatenate(q_opt_30fps).tolist()
        new_qc_opt.append(q_opt_30fps)
        
    if Check.Rviz == True:
                
        print('')
        print('############## Rviz simulation ############## ')
    
        ta.talker(new_qc_opt)  
    
    lbTe = [];
    ubTe = [];
    ubtq = [];
    lbtq = [];
    #Save the csv file
    SaveCsv(folder,qc_opt,qcd_opt,F_opt_loc,tau_LR,tau_RR,lbtq,ubtq,box_pos,box_rpy,Tw,lbTe,ubTe)
        
print('Done')    

        



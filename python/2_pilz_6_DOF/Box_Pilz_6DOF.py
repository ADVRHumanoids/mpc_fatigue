# -*- coding: utf-8 -*-
"""
Created by Pasquale Buonocore 30/10/2019 @IIT

Two pilz robots try keep a box in its initial position.
The torque constraints changes in time.
Look at Mpc fatigue meeting slides of the 06/11/2019 to have a mathematical
problem description.


If you set Check.RightConst = True the right robot joints 0-1-2 will be constrained.
If you set Check.LeftConst = True the left robot joints 0-1 will be constrained.
If you set Check.OCP = True the optimization problem will be solved
If you set Check.Plot = True the solution will be plotted


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
   TorqueConstraint = True
   LeftConst = False
   RightConst = True
   Plot = True

###############################################################################
# ------------------------------ Functions -----------------------------------#
###############################################################################

if Check.Plot == True or Check.OCP == True:
    
#----------------------------  Load the urdf  --------------------------------#
    pilzLR = rospy.get_param('/robot_description_1')
    pilzRR = rospy.get_param('/robot_description_2')
    end_point = 'end_effector'
    
    # Forward kinematics
    fk_LR = casadi.Function.deserialize(pin.generate_forward_kin(pilzLR, end_point))
    fk_RR = casadi.Function.deserialize(pin.generate_forward_kin(pilzRR, end_point))

    # Jacobians
    jac_LR = casadi.Function.deserialize(pin.generate_jacobian(pilzLR, end_point))
    jac_RR = casadi.Function.deserialize(pin.generate_jacobian(pilzRR, end_point))
    
    # Inverse dynamics
    Idyn_LR = casadi.Function.deserialize(pin.generate_inv_dyn(pilzLR))
    Idyn_RR = casadi.Function.deserialize(pin.generate_inv_dyn(pilzRR))


###############################################################################
# --------------------------- Initial conditions -----------------------------#
###############################################################################

    # Define time 
    T = 2.
    N = 50
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
    
    Lbox = 0.2
    #Initial position end effector 1
    x_ini_LR = 0.2
    y_ini_LR = 0.6
    z_ini_LR = 0.4
    p_init_LR = np.array([x_ini_LR,y_ini_LR,z_ini_LR])
    
    #Initial posizione end effector 2
    x_ini_RR = x_ini_LR + Lbox
    y_ini_RR = y_ini_LR
    z_ini_RR = z_ini_LR
    p_init_RR = np.array([x_ini_RR,y_ini_RR,z_ini_RR])
    
    # Set the box final position
    x_box_des = (x_ini_LR + x_ini_RR)/2
    y_box_des = (y_ini_LR + y_ini_RR)/2
    z_box_des = (z_ini_LR + z_ini_RR)/2
    #z_box_des = 0.0
    
    L = dot(p_init_LR - p_init_RR, p_init_LR - p_init_RR)
    p_des = np.array([x_box_des,y_box_des,z_box_des])
    
###############################################################################
# ------------------------ Solve forward kinematics --------------------------#
###############################################################################
    
    print('### Inverse kinematic solution ###')
# ----------------------- Solve forward kinematics pilz_1 --------------------#
    #From the desider initial condition, compute the inverse kinematic to determine
    pos_LR = fk_LR(q = qc_f)['ee_pos']
    pos_des_LR = SX([x_ini_LR,y_ini_LR,z_ini_LR])
    
    rot_ref = np.matrix("0 0 1; 0 1 0 ; -1 0 0")
    rot_LR = fk_LR(q = qc_f)['ee_rot']
    
    des = 1000*dot(pos_LR - pos_des_LR, pos_LR - pos_des_LR)
    des += dot (rot_LR - rot_ref, rot_LR - rot_ref)
    #Nlp problem
    prob_LR = dict(x = qc_f, f = des )
    #Create solver interface
    solver_LR = nlpsol('solver','ipopt', prob_LR)
    sol_LR = solver_LR(x0 = [0.0,0.0,0.0,0.0,0.0,0.0])
    
# --------------------- Solve forward kinematics pilz_2 ----------------------#
    #From the desider initial condition, compute the inverse kinematic to determine
    pos_RR = fk_RR(q = qc_s)['ee_pos']
    pos_des_RR = SX([x_ini_RR,y_ini_RR,z_ini_RR])
    
    rot_ref = np.matrix("0 0 -1; 0 1 0 ; 1 0 0")
    rot_RR = fk_RR(q = qc_s)['ee_rot']
    
    
    des = 1000*dot(pos_RR - pos_des_RR, pos_RR - pos_des_RR)
    des += dot (rot_RR - rot_ref, rot_RR - rot_ref)
    #Nlp problem
    prob_RR = dict(x = qc_s, f = des)
    #Create solver interface
    solver_RR = nlpsol('solver','ipopt', prob_RR)
    sol_RR = solver_RR(x0 = [0.0,0.0,0.0,0.0,0.0,0.0])
    
    #Initial position pilz 1 and pilz 2
    qc_f_init = sol_LR['x']
    qc_s_init = sol_RR['x']
    #Visualize initial condition
    ti.talker(qc_f_init,qc_s_init)
    
    #Robots initial condition
    qc_init = vertcat(qc_f_init,qc_s_init)
    qcdot_init = np.zeros(2*nq).tolist()
    qcddot_init = np.zeros(2*nq).tolist()
    

# ------------------------------- OCP PROBLEM --------------------------------#


if Check.OCP == True:
    print('')
    print('############## Optimal control problem ############## ')
    
    #   BOUNDS 
     
    # Joint angles bound
    lj = np.array([-2.96,-2.53,-2.35,-2.96,-2.96,-3.12,-2.96,-2.53,-2.35,-2.96,-2.96,-3.12])
    uj = np.array([2.96,2.53,2.35,2.96,2.96,3.12,2.96,2.53,2.35,2.96,2.96,3.12])
    # Joint velocity bound
    lbqdot =  - 1.0 # 2.0 # [ rad/s ]
    ubqdot =  0.5 #2.0  # [ rad/s ]
    # Mass
    m = 30 # [ Kg ]
    #Tollerance
    pos_toll = 0.0001 
    
    #   EMPTY NLP 
    
    w = []
    lbw = []
    ubw = []
    g = []
    lbg = []
    ubg = []
    J = 0
    
    #   Define weight of the box
    Fdes = SX(9.81 * m)
    
    #Bounds
    tau_RR0 = []
    tau_RR1 = []
    tau_RR2 = []
    

    tau_LR0 = []
    tau_LR1 = []
    

#--------------------------- FILL THE NLP IN ---------------------------------#

    
    
    # 6 states each robot
    qc_k = SX.sym('qc0', 2*nq)
    w.append(qc_k)
    lbw +=  [qc_init] # .tolist()
    ubw +=  [qc_init] # .tolist() 
    
    for k in range(N):
        
        ###################################################   
        ##                   CONTROLS                    ##   
        ###################################################
        
        #Control at each interval
        qcdname = 'qcd'+ str(k)
        qcd_k = SX.sym(qcdname, 2*nq)
        w.append(qcd_k)
        if k == 0:
            lbw +=  qcdot_init
            ubw +=  qcdot_init
        else:
            lbw +=  np.full((1,2*nq),lbqdot)[0].tolist() 
            ubw +=  np.full((1,2*nq),ubqdot)[0].tolist()
        
        ###################################################   
        ##   NEW POSITION, FORCE AND APPLICATION POINT   ##   
        ###################################################
        
        # Define E1 and E2 position at each step
        E1 = fk_LR( q = qc_k[0:6] )['ee_pos']
        E2 = fk_RR( q = qc_k[6:12])['ee_pos']
        
        # Gravitational force application point
        pbox = (E1 + E2)/2
                 
        #Force at the left end effector
        moment_component = SX.zeros(3)
        F_name = 'FLR'+ str(k)
        F_LR = SX.sym(F_name,3) 
        w.append(F_LR)
        lbw +=  np.full((1,F_LR.shape[0]),-np.inf)[0].tolist() 
        ubw +=  np.full((1,F_LR.shape[0]),np.inf)[0].tolist() 
        W_LR = vertcat(F_LR,moment_component)
        
        #Force at the right end effector
        F_name = 'FRR'+ str(k)
        F_RR = SX.sym(F_name,3) 
        w.append(F_RR)
        lbw +=  np.full((1,F_RR.shape[0]),-np.inf)[0].tolist() 
        ubw +=  np.full((1,F_RR.shape[0]),np.inf)[0].tolist() 
        W_RR = vertcat(F_RR,moment_component)
        
        ###################################################   
        ##                 CONSTRAINTS                   ##   
        ###################################################
        
        #Force static equilibrium
        g.append(vertcat( F_LR[2] + F_RR[2] - Fdes, F_LR[0] + F_RR[0], F_LR[1] + F_RR[1]))
        lbg +=  np.full((1,3),-pos_toll)[0].tolist() 
        ubg +=  np.full((1,3),pos_toll)[0].tolist() 
        
        #Moment static equilibrium
        g.append(vertcat(cross(E1 - E2,F_LR) + cross(E2 - E1,F_RR)))

        lbg +=  np.full((1,3),-pos_toll)[0].tolist() 
        ubg +=  np.full((1,3),pos_toll)[0].tolist() 
        
        #Distance between end effectors
        g.append( dot(E1 - E2, E1 - E2) - L)
        lbg += np.zeros(1).tolist()  
        ubg += np.zeros(1).tolist()
      
        
        if Check.TorqueConstraint == True:
                #Torque limits
                J_LR = jac_LR(q = qc_k[0:6])["J"]
                J_RR = jac_RR(q = qc_k[6:12])["J"]
                tauLR = Idyn_LR(q=qc_k[0:6], qdot= qcd_k[0:6], qddot = qcddot[0:6])['tau'] - mtimes(J_LR.T,W_LR)
                tauRR = Idyn_RR(q=qc_k[6:12], qdot= qcd_k[6:12], qddot = qcddot[6:12])['tau'] - mtimes(J_RR.T,W_RR)
                
                
                if Check.RightConst == True:
                    
                    g.append(vertcat(tauRR[3:6]))
                    lbg += np.full((1,3),-500.0)[0].tolist()   
                    ubg += np.full((1,3),500.0)[0].tolist()
                    
                    #RIGHT ROBOT
                    g.append(tauRR[0])
                    if k < int(N/3):
                        lbg += [-200.0] 
                        ubg += [100.0]
                        tau_RR0.append(-200.0)
                        tau_RR0.append(100.0)
                    elif int(N/3) <= k < int(2*N/3):
                        lbg += [-50.0] 
                        ubg += [50.0]
                        tau_RR0.append(-50.0)
                        tau_RR0.append(50.0)
                    else:
                        lbg += [-5.0] 
                        ubg += [5.0]
                        tau_RR0.append(-5.0)
                        tau_RR0.append(5.0)
                        
                    g.append(tauRR[1])
                    if k < int(N/3):
                        lbg += [-60.0] 
                        ubg += [60.0]
                        tau_RR1.append(-60.0)
                        tau_RR1.append(60.0)
                    elif int(N/3) <= k < int(2*N/3):
                        lbg += [-30.0] 
                        ubg += [10.0]
                        tau_RR1.append(-30.0)
                        tau_RR1.append(10.0)
                    else:
                        ubg += [5.0]
                        lbg += [-5.0]
                        tau_RR1.append(-5.0)
                        tau_RR1.append(5.0)
    
                    g.append(tauRR[2])
                    if k < int(N/3):
                        lbg += [-30.0] 
                        ubg += [40.0]
                        tau_RR2.append(-30.0)
                        tau_RR2.append(40.0)
                    elif int(N/3) <= k < int(2*N/3):
                        lbg += [-20.0] 
                        ubg += [20.0]
                        tau_RR2.append(-20.0)
                        tau_RR2.append(20.0)
                    else:
                        lbg += [-10.0] 
                        ubg += [5.0]
                        tau_RR2.append(-10.0)
                        tau_RR2.append(5.0)
                else:
                    g.append(vertcat(tauRR))
                    lbg += np.full((1,6),-500.0)[0].tolist()   
                    ubg += np.full((1,6),500.0)[0].tolist()
                    
                    
                if Check.LeftConst == True:
                    
                    g.append(vertcat(tauLR[2:6]))
                    lbg += np.full((1,4),-500.0)[0].tolist()   
                    ubg += np.full((1,4),500.0)[0].tolist()
                    #LEFT ROBOT
                    g.append(tauLR[0])
                    if k < int(N/3):
                        lbg += [-600.0] 
                        ubg += [400.0]
                        tau_LR0.append(-600.0)
                        tau_LR0.append(400.0)
                    elif int(N/3) <= k < int(2*N/3):
                        lbg += [-50.0] 
                        ubg += [50.0]
                        tau_LR0.append(-50.0)
                        tau_LR0.append(50.0)
                    else:
                        lbg += [-5.0]
                        ubg += [5.0]
                        tau_LR0.append(-0.0)
                        tau_LR0.append(5.0)
                        
                    g.append(tauLR[1])
                    if k < int(N/3):
                        lbg += [-60.0] 
                        ubg += [60.0]
                        tau_LR1.append(-60.0)
                        tau_LR1.append(60.0)
                    elif int(N/3) <= k < int(2*N/3):
                        lbg += [-30.0] 
                        ubg += [40.0]
                        tau_LR1.append(-30.0)
                        tau_LR1.append(40.0)
                    else:
                        lbg += [-5.0]
                        ubg += [5.0]
                        tau_LR1.append(-5.0)
                        tau_LR1.append(5.0)
                        
                else:
                    g.append(vertcat(tauLR))
                    lbg += np.full((1,6),-500.0)[0].tolist()   
                    ubg += np.full((1,6),500.0)[0].tolist()

          
        # Orientation condition
        # I want the end effector to look each other
        rot_E1 = fk_LR(q = qc_k[0:6])['ee_rot']
        rot_E2 = fk_RR(q = qc_k[6:12])['ee_rot']
        
#        g.append(dot(rot_E1[0::3], rot_E2[1::3]) ) # + SX(1.0))
#        lbg +=  np.full((1,1),-pos_toll)[0].tolist() 
#        ubg +=  np.full((1,1),pos_toll)[0].tolist() 

        ###################################################   
        ##                 COST FUNCTION                 ##   
        ###################################################
        
        J += 100*dot(pbox - p_des , pbox - p_des)
        J += mtimes(qcd_k.T,qcd_k)
        #J += 1000*dot(rot_E1[2::3], rot_E2[2::3]) + SX(1.0)
        
        ###################################################   
        ##                 NEW VARIABLES                 ##   
        ###################################################
        
        #Integration
        q_next = qc_k + qcd_k * h
        
        #New local state
        qname = 'qc' + str(k+1)
        qc_k= SX.sym(qname,2*nq)
        w.append(qc_k)
        lbw += lj.tolist()
        ubw += uj.tolist()
            
        #Continuity constraint
        g.append(q_next - qc_k)
        lbg +=  np.zeros(2*nq).tolist() 
        ubg +=  np.zeros(2*nq).tolist()
        
    ubg = vertcat(*ubg)
    lbg = vertcat(*lbg)
    ubw = vertcat(*ubw)
    lbw = vertcat(*lbw)
    g = vertcat(*g)
    w = vertcat(*w)
    
    
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
    
    print('')
    print('############## Creating csv file ############## ')
    
    #Save the solution in a CSV file
    with open('/home/user/workspace/src/mpc_fatigue/plotter/solution.csv', 'wb') as myfile:
        wr = csv.writer(myfile)
        wr.writerow(sol)
        
###############################################################################
#-------------------------------- PLOTTING -----------------------------------#
###############################################################################
if Check.Plot == True or Check.OCP == True:
    
    sol = []
    
    with open('/home/user/workspace/src/mpc_fatigue/plotter/solution.csv', 'rb') as solution:
        for line in solution:
            x = line.split(',')
            for val in x:
                sol.append(float(val))
                
    nf = 3 # force component
    nq = 6
    n = 4*nq + 2*nf # element solution of each step
    narm = 2 
    
    #Empty lists
    qc_opt = []
    qcd_opt = []
    F_opt = []
    tau_LR = []
    tau_RR = []
    
    #Extract variables
    k = 0
    while (k < N):
        qc_opt.append(sol[k*n:k*n+2*nq])
        qcd_opt.append(sol[k*n+2*nq:k*n+4*nq])
        F_opt.append(sol[k*n+4*nq:k*n+n])
        J_LR = jac_LR(q = sol[k*n:k*n+2*nq][0:nq])["J"]
        J_RR = jac_RR(q = sol[k*n:k*n+2*nq][nq:2*nq])["J"]
        Inv_dLR = Idyn_LR(q = sol[k*n:k*n+2*nq][0:nq], qdot = sol[k*n+2*nq:k*n+4*nq][0:nq],qddot = qcddot_init[0:nq])['tau']
        Inv_dRR = Idyn_RR(q = sol[k*n:k*n+2*nq][nq:2*nq], qdot = sol[k*n+2*nq:k*n+4*nq][nq:2*nq],qddot = qcddot_init[nq:2*nq])['tau']
        WLR = vertcat(sol[k*n+4*nq:k*n+n][0:nf],np.zeros(3).tolist())
        WRR = vertcat(sol[k*n+4*nq:k*n+n][nf:2*nf],np.zeros(3).tolist())
        tau_LR.append( Inv_dLR - mtimes(J_LR.T,WLR))
        tau_RR.append( Inv_dRR - mtimes(J_RR.T,WRR))
        k+= 1
        
    #Make a unique list
    qc_opt = np.concatenate(qc_opt).tolist()
    qcd_opt = np.concatenate(qcd_opt).tolist()
    F_opt = np.concatenate(F_opt).tolist()
    tau_LR = np.concatenate(tau_LR).tolist()
    tau_RR = np.concatenate(tau_RR).tolist()

#------------------- JOINT ANGLES AND JOINT VELOCITIES PLOTS -----------------#
if Check.Plot == True :   
    pwd = '/home/user/workspace/src/mpc_fatigue/plotter/figure_folder/'
    form = 'png' #or svg
    print('')
    print('############## Creating and saving images ############## ')
    
    rob = ['Left','Right']
    for i in range(1,3):
        #if i == 2 plots velocities
        #if i == 1 plots angles
        for r in range(1,narm+1):
            robot = r
            if (i == 2): 
                r = r + i
                plot = qcd_opt
                kind = 'velocities'
            else:
                plot = qc_opt
                kind = 'angles'
            plt.figure(r)
            plt.suptitle(rob[i-1] + ' robot '+ ' joint '+ kind)
            #Plot all the states
            for j in range(nq): 
                plt.subplot(2,nq/2,j+1)
                plt.title('Joint ' + str(j))
                if (robot == narm): j = j + nq
                plt.step(tgrid,plot[j::2*nq],'-')
                plt.ylim(-2.0,2.0)
                plt.grid()
            plt.savefig(pwd + str(robot) + '_robot_'+ '_joint' + kind + '.' + form, format=form)
            
            
#----------------------------- TORQUE PLOTS ----------------------------------#    
    
    for u in range(1,narm+1):
        plt.figure(u+r)
        
        if u == 1:
            plot = tau_LR
            robot = 'Left'
        else:
            plot = tau_RR
            robot = 'Right'
            
        plt.suptitle(str(robot) + ' robot '+ 'torques' )
        #Plot all the torques
        for j in range(nq): 
            plt.subplot(2,nq/2,j+1)
            plt.title('Joint ' + str(j) + ' torque')
            plt.step(tgrid,plot[j::nq],'-')
            plt.ylim(-60.0,60)
            if Check.OCP == True:
                if u == 2:
                    if j == 0 and tau_RR0 != []:
                        plt.plot(tgrid,tau_RR0[0::2],'--')
                        plt.plot(tgrid,tau_RR0[1::2],'--')
                    elif j == 1 and tau_RR1 != []:
                        plt.plot(tgrid,tau_RR1[0::2],'--')
                        plt.plot(tgrid,tau_RR1[1::2],'--')
                    elif j == 2 and tau_RR2 != []:
                        plt.plot(tgrid,tau_RR2[0::2],'--')
                        plt.plot(tgrid,tau_RR2[1::2],'--')
                else:
                    if j == 0 and tau_LR0 != []:
                        plt.plot(tgrid,tau_LR0[0::2],'--')
                        plt.plot(tgrid,tau_LR0[1::2],'--')
                    elif j == 1 and tau_LR1 != []:
                        plt.plot(tgrid,tau_LR1[0::2],'--')
                        plt.plot(tgrid,tau_LR1[1::2],'--')
                    
            plt.grid()
        
        plt.savefig(pwd + str(robot)+ '_robot_'+ 'joint_torque.' + form, format = form)
            
    
#------------------------------ FORCE PLOTS ----------------------------------#   
    
    axis = 'xyz'
    for f in range(1,narm+1):
        plt.figure(u+r+f)
        if f == 1:
            robot = 'Left'
        else:
            robot = 'Right'
            
        plt.suptitle( str(robot) + ' robot ' + ' end effector force')
        for j in range(nq/2):
            plt.subplot(nq/2,1,j+1)
            plt.title('Force ' + axis[j])
            if (f == narm): j = j + nq/2
            plt.step(tgrid,F_opt[j::nq],'-')
            plt.ylim(-5.0,200.0)
            #plt.autoscale(True, 'y', tight = True)
            plt.grid()
        
        plt.savefig(pwd + str(robot)+ '_robot_'+ 'force.' + form, format = form)
            
    
            
    print('')
    print('############## Rviz simulation ############## ')
    ta.talker(qc_opt,nq)  
    #plt.show()
    
    print('ciao')
        
        
        
        
        
        
        
        
        
        
        
        
        
        
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
from Centauros_features import *
import rospy
import mpc_fatigue.pynocchio_casadi as pin
import numpy as np
import Centauros_inverse_kinematics_talker as ti
import Centauro_dynamics_talker as ta
import csv
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

###############################################################################
#--------------------------What do you want to do? ---------------------------#
###############################################################################

class Check:
   OCP = True
   TorqueConstraint = True
   LeftConst =  False
   RightConst = True
   Plot = False
   Rviz = True
   RealTime = False
   
   '''
   #### IT WORKS ####
   Const1 = True will use the FULL POSITION CONSTRAINTS
   
   
   #### IT WORKS ####
   Const2 = True will use the FULL VELOCITY CONSTRAINTS
   
   
   #### IT WORKS ####
   Const3 = True will use the HYBRID CONSTRAINTS: 3 POSITIONS + 3 ANGULAR VELOCITIES
   
   
   #### IT WORKS ####
   Const4 = True will use the HYBRID CONSTRAINTS: 3 LINEAR VELOCITIES + 3 ORIENTATION ANGLES
   '''
   Const1 = False
   Const2 = True
   Const3 = False
   Const4 = False
   
# Automatically defines the folders where to store images and csv files   
if Check.Const1 == True: folder = "FullPosition/"
if Check.Const2 == True: folder = "FullVelocity/"
if Check.Const3 == True: folder = "Pos_angVel/"
if Check.Const4 == True: folder = "Lvel_Angles/"
   
if  Check.LeftConst == True and Check.RightConst == True: subfolder = "Both/"
   
if  Check.LeftConst == True and Check.RightConst == False: subfolder = "Left/"
   
if  Check.LeftConst == False and Check.RightConst == True: subfolder = "Right/"

fold = folder + subfolder

###############################################################################
# ------------------------ Solve forward kinematics --------------------------#
###############################################################################
    
print('### Inverse kinematic solution ###')

# General parameters
nq = 14
qc = SX.sym('qc', nq)
g = []
lbg = []
ubg = []

#Initial positions
Lbox = 0.4
x_ini = 0.9
y_ini = Lbox/2
z_ini = 1.3

#From the desider initial condition, compute the inverse kinematic to determine

pos_LA = ForwKinLA(qc,'pos')
pos_RA = ForwKinRA(qc,'pos')
rot_LA = ForwKinLA(qc,'rot')
rot_RA = ForwKinRA(qc,'rot')

pos_des_LA = SX([x_ini,y_ini,z_ini])
pos_des_RA = SX([x_ini,-y_ini,z_ini])
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
print("Centauro initial condition is: " + str(qc_0))
print(" ")
#Visualize initial condition
ti.talker(qc_0)
print("")
    
    
###############################################################################
# --------------------------- INITIAL CONDITIONS -----------------------------#
###############################################################################

#Check initial condition
print("####### Checking initial condition ###### ")
print("")

for i in range(14):
    if i == 0: 
        print('Left arm constraints:')
        print("")
        
    print(str(Centauro_features.joint_angle_lb[i]) + ' < ' + str(qc_0[i]) + ' < ' +  str(Centauro_features.joint_angle_ub[i]))
    
    if (Centauro_features.joint_angle_lb[i] > qc_0[i]) or (Centauro_features.joint_angle_ub[i] < qc_0[i] ):
        raise Exception('Invalid initial condition. Bounds exceeded')
        
    if i == 6: 
        print("")
        print("Right arm constraints:")
        print("")
        
#Define the initial position of the two end effectors
posLA = ForwKinLA(qc_0,'pos')
posRA = ForwKinRA(qc_0,'pos')           

L = dot(posLA - posRA, posLA - posRA)

#Initial posizion box
x_ini_box = posLA[0]
y_ini_box = posLA[1] + posRA[1]
z_ini_box = (posLA[2] + posRA[2])/2

# Set the box final position
x_box_des = x_ini_box
y_box_des = y_ini_box
z_box_des = z_ini_box

p_des = np.array([x_box_des,y_box_des,z_box_des])

#Plot the initial condition in Rviz 
print(" ")
print("############## The initial condition is set in Rviz ##############")
print(" ")
print("Box initial condition :" + str([x_ini_box,y_ini_box,z_ini_box]))


###############################################################################
#-------------------- COMPUTE INITIAL RELATIVE POSITION ----------------------#
###############################################################################
    
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
dist = []
dist.append(pos_1in2)


###############################################################################
#-------------------- COMPUTE INITIAL ORIENTATION ERROR ----------------------#
###############################################################################

R_o = mtimes(rot1,rot2.T)
#Compute the skew metrix of R_o
R_skew = (R_o - R_o.T)/2

ex = np.round(R_skew[2,1],3)[0][0]
ey = np.round(R_skew[2,0],3)[0][0]
ez = np.round(R_skew[1,0],3)[0][0]

e_des = np.array([ex,ey,ez]).reshape(3,1)


###############################################################################
# -------------------------- START OCP PROBLEM -------------------------------#
###############################################################################

 # Define time 
T = 2.
N = 50
h = T/N

# Define a certain time grid
tgrid = [T/N*k for k in range(N)]

#Variablesinv(mat)
nq = 14

#Same zero acceleration and velocity for both
qc_dot0 = np.zeros(nq).tolist() # Joint acceleration
qc_ddot0 = np.zeros(nq).tolist() # Joint acceleration
    
#BOUNDS 
 
# Joint angles bound
lj = Centauro_features.joint_angle_lb # [rad]
uj = Centauro_features.joint_angle_ub # [rad]

# Joint velocity bound
lbqdot =  - Centauro_features.joint_velocity_lim # [rad/s]
ubqdot = Centauro_features.joint_velocity_lim # [rad/s]

#Torque bounds
lbtorque = -Centauro_features.joint_torque_lim # [N/m]
ubtorque = Centauro_features.joint_torque_lim # [N/m]

# Mass to lift up
m = 10 # [ Kg ]

#Tollerance
toll = 0.001 # [N] on [Nm]

lbt = []
ubt = []


    
if Check.OCP == True:
    print('')
    print('############## Optimal control problem ############## ')
    
    
    #   EMPTY NLP 
    w = []
    lbw = []
    ubw = []
    g = []
    lbg = []
    ubg = []
    J = 0
    Orient = []
    #Box weight
    Fdes = SX(9.81 * m)
    
    
    # 7 states each robot
    qc_k = SX.sym('qc0', nq)
    w.append(qc_k)
    lbw +=  [qc_0] # .tolist()
    ubw +=  [qc_0] # .tolist() 
    
    
    for k in range(N):
            
        ###################################################   
        ##                   CONTROLS                    ##   
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
        ##   NEW POSITION, FORCE AND APPLICATION POINT   ##   
        ###################################################
        
        # Define E1 and E2 position at each step
        E1 = ForwKinLA(qc_k,'pos')
        E2 = ForwKinRA(qc_k,'pos')
        
        # Gravitational force application point
        pbox = (E1 + E2)/2
                 
        #Force at the left end effector
        moment_component = SX.zeros(3)
        F_name = 'FLR'+ str(k)
        F_LR = SX.sym(F_name,3) 
        w.append(F_LR)
        lbw +=  np.full((1,F_LR.shape[0]),-np.inf)[0].tolist() 
        ubw +=  np.full((1,F_LR.shape[0]),np.inf)[0].tolist() 
        W_LA = vertcat(F_LR,moment_component)
        
        #Force at the right end effector
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
        g.append(vertcat(cross(E1 - E2,F_LR) + cross(E2 - E1,F_RR)))
        lbg +=  np.full((1,3),-toll)[0].tolist() 
        ubg +=  np.full((1,3),toll)[0].tolist() 
    
        # FULL POSITION CONSTRAINT
        if Check.Const1 == True or Check.Const3 == True or Check.Const4 == True : 
            
            pos1 = ForwKinLA(qc_k,'pos')
            rot1 = ForwKinLA(qc_k,'rot')
            pos1 = vertcat(pos1,SX(1.0))
            
            #E2 effector    
            pos2 = ForwKinRA(qc_k,'pos')
            rot2 = ForwKinRA(qc_k,'rot' )
            
            if Check.Const1 == True or Check.Const3 == True:
                #### RELATIVE POSITION ####
                mat_inv = vertcat(horzcat(rot2.T,-mtimes(rot2.T,pos2)),SX([0.0,0.0,0.0,1.0]).T)
                #Compute the relative position of the E1 in E2
                pos_1in2 = mtimes(mat_inv,pos1)
                
                #Add the constraint - The relative position should be the same
                g.append(pos_1in2[0:3] - dist[k][0:3])
                lbg += np.zeros(3).tolist()  
                ubg += np.zeros(3).tolist()
                dist.append(pos_1in2)
            
            if Check.Const1 == True or Check.Const4 == True:
                #### RELATIVE ORIENTATION ####
                R_o = mtimes(rot1,rot2.T)
                #Compute the skew metrix of R_o
                R_skew = (R_o - R_o.T)/2
                #Extract errors
                ex = R_skew[2,1]
                ey = R_skew[2,0]
                ez = R_skew[1,0]
                e_o = np.array([ex,ey,ez])
                
                g.append(reshape(e_o,(3,1)) - e_des)
                lbg += np.zeros(3).tolist()  
                ubg += np.zeros(3).tolist()

        # FULL VELOCITY CONSTRAINT
        if Check.Const2 == True or Check.Const3 == True or Check.Const4 == True:
            
            # 6 relative velocity constraints
            JLA = Jac_LA(qc_k)
            JRA = Jac_RA(qc_k)
            
            # 3 relative position constraints
            pos1 = ForwKinLA(qc_k,'pos')
            rot1 = ForwKinLA(qc_k,'rot')
            
            #E2 effector    
            pos2 = ForwKinRA(qc_k,'pos')
            rot2 = ForwKinRA(qc_k,'rot' )
            pos2 = vertcat(pos2,SX(1.0))
            #Matrice di trasformazione dal sistema di riferimento 0 al sistema di rifemento 1
            mat_inv = vertcat(horzcat(rot1.T,-mtimes(rot1.T,pos1)),SX([0.0,0.0,0.0,1.0]).T)
            #Compute the relative position of the E2 in E1
            p = mtimes(mat_inv,pos2)
            #From pos 2 in 1 I can compute the skew matrix
            I = np.array([1,0,0,0,1,0,0,0,1]).reshape(3,3)
            Sp2in1 = np.array([0, -p[2], p[1],p[2],0,-p[0],-p[1], p[0],0]).reshape(3,3)
            r0E1 = rot1.T
            zerome = np.zeros(9).reshape(3,3)
            
            #IMPORTANT PSI MATRIX
            psi = vertcat(horzcat(I, - Sp2in1), horzcat(zerome, I))
            
            #IMPORTANT ROTATION MATRIX
            omega = vertcat(horzcat(r0E1, zerome), horzcat(zerome, r0E1))
            
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
                

        if Check.TorqueConstraint == True:
                #Torque limits
                JLA = Centauro_features.jac_la(qc_k)
                JRA = Centauro_features.jac_ra(qc_k)
                tau = InvDyn(qc_k, qcd_k, qc_ddot0) + mtimes(JLA.T,W_LA) + mtimes(JRA.T,W_RA)
                
                #IF BOTH RIGHT AND LEFT ARM ARE JUST CONSTRAINED TO HIGH VALUE
                if Check.RightConst == False and Check.LeftConst == False:
                    g.append(tau)
                    lbg += [lbtorque]
                    ubg += [ubtorque]
                    lbt.append(lbtorque)
                    ubt.append(ubtorque)
                    
                #IF THE LEFT ARM IS CONSTRAINED AND THE RIGHT ONE CONSTRAINED TO HIGH VALUE
                if Check.LeftConst == True and Check.RightConst == False:
                    
                    g.append(tau[0:3])
                    
                    if k < int(N/3):
                        lbg += [lbtorque[0:3]]
                        ubg += [ubtorque[0:3]]
                        lbt.append(lbtorque[0:3])
                        ubt.append(ubtorque[0:3])
                    else:
                        lbg += np.full((1,3),-5.0)[0].tolist()
                        ubg += np.full((1,3),5.0)[0].tolist()
                        lbt.append(np.full((1,3),-5.0)[0])
                        ubt.append(np.full((1,3),5.0)[0])

                    #Constraint the remaining one
                    g.append(tau[3:])
                    lbg += [lbtorque[3:]]
                    ubg += [ubtorque[3:]]
                    lbt += [lbtorque[3:]]
                    ubt += [ubtorque[3:]]
                
                #IF THE RIGHT ARM IS CONSTRAINED AND THE LEFT ONE IS CONSTRAINED TO HIGH VALUE
                elif Check.LeftConst == False and Check.RightConst == True:
                    
                    #Constraint the left one to high value
                    g.append(tau[0:nq/2])
                    lbg += [lbtorque[0:nq/2]]
                    ubg += [ubtorque[0:nq/2]]
                    lbt.append(lbtorque[0:nq/2])
                    ubt.append(ubtorque[0:nq/2])
                    
                    
                    g.append(tau[nq/2:nq/2+3])
                    
                    if k < int(N/3):
                        lbg += [lbtorque[nq/2:nq/2+3]]
                        ubg += [ubtorque[nq/2:nq/2+3]]
                        lbt.append(lbtorque[nq/2:nq/2+3])
                        ubt.append(ubtorque[nq/2:nq/2+3])
                    else:
                        lbg += np.full((1,3),-5.0)[0].tolist()
                        ubg += np.full((1,3),5.0)[0].tolist()
                        lbt.append(np.full((1,3),-5.0)[0])
                        ubt.append(np.full((1,3),5.0)[0])
                        
                    #Constraint the remaining one
                    g.append(tau[nq/2+3:])
                    lbg += [lbtorque[nq/2+3:]]
                    ubg += [ubtorque[nq/2+3:]]
                    lbt.append(lbtorque[nq/2+3:])
                    ubt.append(ubtorque[nq/2+3:])
                    
                    
                #IF BOTH THE THE RIGHT AND LEFT ARM ARE CONSTRAINED
                elif Check.RightConst == True and Check.LeftConst == True:
                    
                    g.append(tau[0:3])
                    
                    if k < int(N/3):
                        lbg += [lbtorque[0:3]]
                        ubg += [ubtorque[0:3]]
                        lbt.append(lbtorque[0:3])
                        ubt.append(ubtorque[0:3])
                    else:
                        lbg += np.full((1,3),-5.0)[0].tolist()
                        ubg += np.full((1,3),5.0)[0].tolist()
                        lbt.append(np.full((1,3),-5.0)[0])
                        ubt.append(np.full((1,3),5.0)[0])

                    #Constraint the remaining one
                    g.append(tau[3:nq/2])
                    lbg += [lbtorque[3:nq/2]]
                    ubg += [ubtorque[3:nq/2]]
                    lbt.append(lbtorque[3:nq/2])
                    ubt.append(ubtorque[3:nq/2])                    
                    
                    
                    g.append(tau[nq/2:nq/2+3])
                    
                    if k < int(N/3):
                        lbg += [lbtorque[nq/2:nq/2+3]]
                        ubg += [ubtorque[nq/2:nq/2+3]]
                        lbt.append(lbtorque[nq/2:nq/2+3])
                        ubt.append(ubtorque[nq/2:nq/2+3])
                    else:
                        lbg += np.full((1,3),-5.0)[0].tolist()
                        ubg += np.full((1,3),5.0)[0].tolist()
                        lbt.append(np.full((1,3),-5.0)[0])
                        ubt.append(np.full((1,3),5.0)[0])
                        
                    #Constraint the remaining one
                    g.append(tau[nq/2+3:])
                    lbg += [lbtorque[nq/2+3:]]
                    ubg += [ubtorque[nq/2+3:]]
                    lbt.append(lbtorque[nq/2+3:])
                    ubt.append(ubtorque[nq/2+3:]) 


        ###################################################   
        ##                 COST FUNCTION                 ##   
        ###################################################
        J += 100*dot(pbox - p_des , pbox - p_des)
        J += mtimes(qcd_k.T,qcd_k)
        
        ###################################################   
        ##                 NEW VARIABLES                 ##   
        ###################################################
        
        #Integration
        q_next = qc_k + qcd_k * h
        
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
        
    ubg = vertcat(*ubg)
    lbg = vertcat(*lbg)
    ubw = vertcat(*ubw)
    lbw = vertcat(*lbw)
    lbt = np.concatenate(lbt)
    ubt = np.concatenate(ubt)
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
    
                                ###################################################   
                                ##                SAVE CSV FILES                 ##   
                                ###################################################
    
    with open('/home/user/workspace/src/mpc_fatigue/Centauro_solutions/' + fold + '/solution.csv', 'wb') as myfile:
        wr = csv.writer(myfile)
        wr.writerow(sol)
        
    with open('/home/user/workspace/src/mpc_fatigue/Centauro_solutions/'+ fold +'/lbt.csv', 'wb') as myfile:
        wr = csv.writer(myfile)
        wr.writerow(lbt)
        
    with open('/home/user/workspace/src/mpc_fatigue/Centauro_solutions/' +fold + '/ubt.csv', 'wb') as myfile:
        wr = csv.writer(myfile)
        wr.writerow(ubt)



if Check.Rviz == True or Check.OCP == True or Check.Plot == True:
    
    sol = []
    lbt = []
    ubt = []
    
    with open('/home/user/workspace/src/mpc_fatigue/Centauro_solutions/' + fold + '/solution.csv', 'rb') as solution:
        for line in solution:
            x = line.split(',')
            for val in x:
                sol.append(float(val))
                
    with open('/home/user/workspace/src/mpc_fatigue/Centauro_solutions/'+ fold +'/lbt.csv', 'rb') as solution:
        for line in solution:
            x = line.split(',')
            for val in x:
                lbt.append(float(val))
                
    with open('/home/user/workspace/src/mpc_fatigue/Centauro_solutions/' +fold + '/ubt.csv', 'rb') as solution:
        for line in solution:
            x = line.split(',')
            for val in x:
                ubt.append(float(val))
                
    nf = 3 # force component
    nq = 14
    n = 2*nq + 2*nf # element solution of each step
    narm = 2 
    
    #Empty lists
    qc_opt = []
    qcd_opt = []
    F_opt = []
    tau_LR = []
    tau_RR = []
    
                                ###################################################   
                                ##               EXTRACT VARIABLE                ##   
                                ###################################################
    
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
        
    #Make a unique list
    qc_opt = np.concatenate(qc_opt).tolist()
    qcd_opt = np.concatenate(qcd_opt).tolist()
    F_opt = np.concatenate(F_opt).tolist()
    tau_LR = np.concatenate(tau_LR).tolist()
    tau_RR = np.concatenate(tau_RR).tolist()
    


if Check.Plot == True :   
    
    pwd = '/home/user/workspace/src/mpc_fatigue/Centauro_solutions/' + fold
    form = 'svg' #or svg#    tau1_1 =np.full((1,N),lbqdot[1])[0].tolist()

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
            plt.suptitle(rob[i-1] + ' arm '+ ' joint '+ kind)
            #Plot all the states
            for j in range(nq/2): 
                plt.subplot(2,4,j+1)
                plt.title('Joint ' + str(j))
                if (robot == narm): j = j + nq/2
                if i == 1:
                    plt.step(tgrid,plot[j::nq],'-')
                    plt.plot(tgrid,np.full((1,N),Centauro_features.joint_angle_lb[j])[0],'--')   
                    plt.plot(tgrid,np.full((1,N),Centauro_features.joint_angle_ub[j])[0],'--')   
                    plt.grid()
                else:
                    plt.step(tgrid,plot[j::nq],'-')
                    plt.plot(tgrid,np.full((1,N),-Centauro_features.joint_velocity_lim[j])[0],'--')   
                    plt.plot(tgrid,np.full((1,N),Centauro_features.joint_velocity_lim[j])[0],'--')   
                    plt.grid()
            plt.savefig(pwd + str(robot) + '_arm_'+ '_joint' + kind + '.' + form, format=form)
            
            
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
        for j in range(nq/2): 
            plt.subplot(2,4,j+1)
            plt.title('Joint ' + str(j) + ' torque')
            plt.step(tgrid,plot[j::nq/2],'-')  
            tau_name = 'tau'+ str(u)+ '_' +str(j)
            if u == 2: j = j + nq/2
            plt.plot(tgrid,lbt[j::nq],'--')   
            plt.plot(tgrid,ubt[j::nq],'--')                       
            plt.grid()
        
        plt.savefig(pwd + str(robot)+ '_arm_'+ 'joint_torque.' + form, format = form)
            
    
#------------------------------ FORCE PLOTS ----------------------------------#   
    
    axis = 'xyz'
    for f in range(1,narm+1):
        plt.figure(u+r+f)
        if f == 1:
            robot = 'Left'
        else:
            robot = 'Right'
            
        plt.suptitle( str(robot) + ' arm ' + ' end effector force')
        for j in range(3):
            plt.subplot(3,1,j+1)
            plt.title('Force ' + axis[j])
            if (f == narm): j = j + 3
            plt.step(tgrid,F_opt[j::6],'-')
            plt.grid()
        
        plt.savefig(pwd + str(robot)+ '_arm_'+ 'force.' + form, format = form)

# ----------------- PLOT BOX POSITIONS AND ORIENTATION -------------------#
    
    k = 0
    box_pos = []
    while k < N :
        E1 = ForwKinLA(sol[k*n:k*n+nq],'pos')
        E2 = ForwKinRA(sol[k*n:k*n+nq],'pos')
        box_pos.append(np.round((E1 + E2)/2,3))
        k = k + 1
        
    box_pos = np.concatenate(box_pos)
    
    fig = plt.figure(u+r+f+2)
    ax = plt.axes(projection="3d")
    ax.scatter3D(np.round(box_pos[0::3],3),np.round(box_pos[1::3],1),np.round(box_pos[2::3],3))
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    
    plt.savefig(pwd + '3Dboxposition.' + form, format = form)
        
   
if Check.Rviz == True:
            
    print('')
    print('############## Rviz simulation ############## ')

    ta.talker(qc_opt,tau_LR,tau_RR,lbt,ubt,tgrid, Check.RealTime)  

print('Done')
    

        
#!/usr/bin/env python
# license removed for brevity

#####################################################
#                    LIBRARIES                      #
#####################################################

# BINDINGS
import mpc_fatigue.pynocchio_casadi as pin

#MY OWN LIBRARIES
from Centauro_functions import *
from Tmodel_library import *

# ROS AND CASADI LIBRARIES
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import JointState
from std_msgs.msg import String 
from std_msgs.msg import Header

from casadi import *
import numpy as np
import rospy
import csv
import time

#####################################################
#           NODE AND TOPIC INITIALIZATION           #
#####################################################

#Define the node name
rospy.init_node('OCPsolver_node') #, anonymous=True)
#Define at which frequency the publisher will publish to the unroller node
rate = rospy.Rate(1000) # 100hz


#####################################################
#           SET PARAMETERS TO SIMULATION            #
#####################################################

class Check:
    Const1 = True               # Const1 = True will use the FULL POSITION CONSTRAINTS 
    Const2 = False              # Const2 = True will use the FULL VELOCITY CONSTRAINTS 
    Const3 = False              # Const3 = True will use the HYBRID CONSTRAINTS: 3 POSITIONS + 3 ANGULAR VELOCITIES 
    Const4 = False              # Const4 = True will use the HYBRID CONSTRAINTS: 3 LINEAR VELOCITIES + 3 ORIENTATION ANGLES 
    MySolver = "ipopt"          # sqpmethod or ipopt'''

#####################################################
#              INITIALIZATION IN ROS                #
#####################################################

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

#####################################################
#               OCP  INITIALIZATION                 #
#####################################################


# Define time and sample nodes
T = 30.
N = 40
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
#Temperature bounds
lbtemp = np.full((1,nq),0.0)[0].tolist()
ubtemp = np.full((1,nq),80.0)[0].tolist()

# COMPUTE INITIAL RELATIVE POSITION AND RELATIVE ERROR

#Left End effector wrt right end effector
RelativePosition_0 = InitialRelativePosition(qc_0)
RelativeOrientation_0 = InitialRelativeOrientationError(qc_0)

#Initialilize the relative position list
RelPosition = [RelativePosition_0]

#Compute the initial torques
WA0 = np.array([0,0,m*9.81/2,0,0,0]).reshape(6,1)

tau0 = InvDyn(qc_0,qc_dot0,qc_ddot0) + vertcat(mtimes(Jac_LA(qc_0).T,WA0),mtimes(Jac_RA(qc_0).T,WA0))
#From the initial torques get the current
T_0 = np.full((1,nq),20.0)[0].tolist() 

#####################################################
#               TOPIC INITIALIZATION                #
#####################################################

#Create the publisher that will send the OPTIMAL TRAJECTORY to the unroller node
#pub = rospy.Publisher('to_unroller_topic', String , queue_size=10)
pub = rospy.Publisher('to_unroller_topic', Float32MultiArray , queue_size=10000)


#INITIAL CONDITION FOR THE FIRST WARMSTART
qc_0 = np.array(qc_0).reshape(1,14)[0].tolist()
F0 = np.array([0,0,m*9.81/2,0,0,m*9.81/2]).reshape(1,6)[0].tolist()
sol0 = qc_0 + T_0 + qc_dot0 + F0
sol0 = sol0 * N + qc_0 + T_0 
#####################################################
#              REAL-TIME OCP ALGORITHM              #
#####################################################

s = 0 #Step numero uno senza IPOPT
ciclo = 0
nf = 3 # force components
n = 3*nq + 2*nf # element solution of each step  
while s < 2:

    print("Solving OCP " + str(s) + " ...")
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
    Fdes = SX(9.81 * m)
    

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
                    
        #Force that the left end effector exerts to the object
        moment_component = SX.zeros(3)
        F_name = 'FLR'+ str(k)
        F_LR = SX.sym(F_name,3) 
        w.append(F_LR)
        lbw +=  np.full((1,3),-np.inf)[0].tolist() 
        ubw +=  np.full((1,3),np.inf)[0].tolist() 
        W_LA = vertcat(F_LR,moment_component)
        
        #Force that the right end effector exerts to the object
        F_name = 'FRR'+ str(k)
        F_RR = SX.sym(F_name,3) 
        w.append(F_RR)
        lbw +=  np.full((1,3),-np.inf)[0].tolist() 
        ubw +=  np.full((1,3),np.inf)[0].tolist() 
        W_RA = vertcat(F_RR,moment_component)
    
        ###################################################  
        ##                 CONSTRAINTS                   ##   
        ###################################################
        
        #Force static equilibrium
        g.append(vertcat( F_LR[2] + F_RR[2] - Fdes, F_LR[0] + F_RR[0], F_LR[1] + F_RR[1]))
        lbg += np.zeros(3).tolist()  
        ubg += np.zeros(3).tolist()
        
        #Moment static equilibrium
        g.append(vertcat(cross(pL - pR,F_LR) + cross(pR - pL,F_RR)))
        lbg += np.zeros(3).tolist()  
        ubg += np.zeros(3).tolist()
    
        # RELATIVE POSE CONSTRAINT
        if Check.Const1 == True or Check.Const3 == True or Check.Const4 == True : 
            
            #pos1 = vertcat(pos1,SX(1.0))
            
            if Check.Const1 == True or Check.Const3 == True:
                #### RELATIVE POSITION ####
                pos_1in2 = mtimes(R01.T,pR) - mtimes(R01.T,pL)
                #Add the constraint - The relative position should be the same
                g.append(pos_1in2[0:3] - RelPosition[k][0:3]) 
#                    lbg +=  np.full((1,3),0.0)[0].tolist() 
#                    ubg +=  np.full((1,3),0.001)[0].tolist() 
                lbg += np.zeros(3).tolist()  
                ubg += np.zeros(3).tolist()
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
                
                
                g.append(reshape(e_k,(3,1)) - RelativeOrientation_0)
                lbg += np.zeros(3).tolist()  
                ubg += np.zeros(3).tolist()
#                    lbg +=  np.full((1,3),0.0)[0].tolist() 
#                    ubg +=  np.full((1,3),0.001)[0].tolist() 

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
#            
        g.append(tau)
        lbg += [lbtorque]
        ubg += [ubtorque]
        lbtq.append(lbtorque)
        ubtq.append(ubtorque)
                
        #Tref = np.full((1,nq),50.0)[0].tolist()      
        ###################################################   
        ##                 COST FUNCTION                 ##   
        ###################################################
        J += 1000*dot(pbox - Box_ini , pbox - Box_ini)
        J += 100*dot(qcd_k,qcd_k)
        J += 10*dot(F_LR,F_LR)
        J += 10*dot(F_RR,F_RR)
        #J += dot(T_k - Tref, T_k - Tref)
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

    if s >= 1:
        opts = {"ipopt": {"warm_start_init_point": "yes","tol": 0.001,"acceptable_tol": 0.001,"constr_viol_tol" : 0.001, "compl_inf_tol": 0.001}}
    else:
        opts = {"ipopt": {"warm_start_init_point": "yes","tol": 0.001,"acceptable_tol": 0.001,"constr_viol_tol" : 0.001, "compl_inf_tol": 0.001}}


    # Allocate a solver
    Solver = nlpsol("solver", Check.MySolver, nlp, opts)

    if s >= 1:
        r = Solver(x0 = sol, lbx = lbw, ubx = ubw, lbg = lbg, ubg = ubg)  
    else:
        r = Solver(x0 = sol0, lbx = lbw, ubx = ubw, lbg = lbg, ubg = ubg) 

    sol = r['x'].full().flatten()

    #Initial condition for the next OCP
    qc_0 = sol[N*n:N*n+nq]
    T_0 = sol[N*n+nq:N*n+2*nq] - 0.05 
    qc_dot0 = sol[k*n+2*nq:k*n+3*nq]

    RelativePosition_0 = np.round(InitialRelativePosition(qc_0),3)
    RelativeOrientation_0 = np.round(InitialRelativeOrientationError(qc_0),3)

    qc_0 = np.round(qc_0,4).tolist()
    T_0 = np.round(T_0,4).tolist()
    qc_dot0 = np.round(qc_dot0,4).tolist()
    
    #Initialilize the relative position list
    RelPosition = [RelativePosition_0]

    print(" Estimated joint temperatures: ")
    print(T_0)

    print("End effector relative position: ")
    print(RelativePosition_0)
        
    print("End effector relative orientation: ")
    print(RelativeOrientation_0)

    
    data_to_send = Float32MultiArray()
    j = 0
    
    while (j < N):
        #Extract the optimal GLOBAL FORCE
        Ftot_global = sol[j*n+3*nq:j*n+n].tolist()
        WrenchL_global = np.array(Ftot_global[0:3] + [0,0,0])
        WrenchR_global = np.array(Ftot_global[3:6] + [0,0,0])
        #Transform FL  in LOCAL FORCE
        RfromWtoL = np.array(ForwKinLA(sol[j*n:j*n+nq],"rot").T)
        HfromWtoL = vertcat(horzcat(RfromWtoL,np.zeros([3,3])),horzcat(np.zeros([3,3]),RfromWtoL))
        F_tot_opt_local = [ mtimes(HfromWtoL,WrenchL_global)[0:3] ]
        
        #Transform FR  in LOCAL FORCE
        RfromWtoR = np.array(ForwKinLA(sol[j*n:j*n+nq],"rot").T)
        HfromWtoR = vertcat(horzcat(RfromWtoR,np.zeros([3,3])),horzcat(np.zeros([3,3]),RfromWtoR))
        F_tot_opt_local.append( mtimes(HfromWtoL,WrenchL_global)[0:3])
        
        data_to_send.data =  np.array(sol[j*n:j*n+nq].tolist() + np.concatenate(np.concatenate(F_tot_opt_local).tolist()).tolist() )
        pub.publish(data_to_send)
        rate.sleep()
        #Move on the counter
        j += 1 
    
        
    print("OCP has been solved, send trajectory to the unroller ...")
    #Upgrade s to let the warmstarting start and the new OCP
    s = s + 1
       

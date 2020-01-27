#!/usr/bin/env python
# license removed for brevity

#####################################################
#                    LIBRARIES                      #
#####################################################

# BINDINGS
import mpc_fatigue.pynocchio_casadi as pin

#MY OWN LIBRARIES
from Centauro_functions import *
from MPC_parameters import *
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
rate = rospy.Rate(mpc.OCP_solver_publish_rate) # 100hz

#####################################################
#              INITIALIZATION IN ROS                #
#####################################################

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
print("##################################")
print("#   Checking initial condition   #")
print("##################################")
print("")
CheckInitialCondition(qc_0)

#####################################################
#               OCP  INITIALIZATION                 #
#####################################################


# Define time and sample nodes
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
toll = mpc.constraint_tollerance     # [N] on [Nm]

# Joint acceleration and velocity initial conditions
qc_dot0 = np.zeros(nq).tolist() 
qc_ddot0 = np.zeros(nq).tolist() 

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
ubtemp = np.full((1,nq),mpc.temperature_bound)[0].tolist()

# COMPUTE INITIAL RELATIVE POSITION AND RELATIVE ERROR

#Left End effector wrt right end effector
RelativePosition_0 = RelativePosition(qc_0)
RelativeOrientation_0 = RelativeOrientationError(qc_0)

#Initialilize the relative position list
RelPosition = [RelativePosition_0]

##Compute the initial torques
#WA0 = np.array([0,0,m*9.81/2,0,0,0]).reshape(6,1)
#
#tau0 = InvDyn(qc_0,qc_dot0,qc_ddot0) + vertcat(mtimes(Jac_LA(qc_0).T,WA0),mtimes(Jac_RA(qc_0).T,WA0))
#print(tau0)
#From the initial torques get the current
T_0 = np.full((1,nq),20.0)[0].tolist() 

BoxPos = mpc.box_initial_position
L = mpc.Lbox
#####################################################
#               TOPIC INITIALIZATION                #
#####################################################

#Create the publisher that will send the OPTIMAL TRAJECTORY to the unroller node

pub = rospy.Publisher('to_unroller_topic', Float32MultiArray , queue_size=10000)

mu = mpc.mu

#INITIAL CONDITION FOR THE FIRST WARMSTART
qc_00 = np.array(qc_0).reshape(1,nq)[0].tolist()
F0 = np.array([0, - m*9.81/(2*mu) ,m*9.81/2,0, m*9.81/(2*mu),m*9.81/2]).reshape(1,6)[0].tolist()
sol0 = qc_00 + T_0 + qc_dot0 + F0
sol0 = sol0 * N + qc_00 + T_0 

#####################################################
#              REAL-TIME OCP ALGORITHM              #
#####################################################

s = 0 #Step numero uno senza IPOPT
ciclo = 0
nf = 3 # force components
n = 3*nq + 2*nf # element solution of each step  


soltot = []
while s < 20:
    print(" ")
    print("##################################")
    print("#         Solving OCP "  + str(s) +"          #")
    print("##################################")
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
                 
        #Force that the left end effector exerts to the object in GLOBAL FRAME
        moment_component = SX.zeros(3)
        F_name = 'FLR'+ str(k)
        F_LR = SX.sym(F_name,3) 
        w.append(F_LR)
        lbw +=  np.full((1,3),-np.inf)[0].tolist() 
        ubw +=  np.full((1,3),np.inf)[0].tolist() 
        W_LA = vertcat(F_LR,moment_component)
        
        #Force that the right end effector exerts to the object in GLOBAL FRAME
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
        # Then it is possible to compute the list of constraint
        # Note that F_C1 is the local force that the enviorment excert on the robot.
        # How do I compute it starting from the GLOBAL FORCE that the robot excert to the enviorment?
        
        #Firstly I change the force reference frame
        F_C1robot_environment_local = mtimes(R01.T,F_LR)
        #Then change the sign to obtaine the force that the enviorment excert on the robot
        F_C1 = - F_C1robot_environment_local
        #Then evaluate the constraint
        C1 = mtimes(A1,F_C1)

        #Friction cone contact 2

        A2 = np.matrix([ [0 , 1 , 0],\
                         [1 , mu , 0 ], \
                         [-1 , mu , 0], \
                         [0 , mu , 1 ], \
                         [0 , mu ,-1 ]])
                         
        #Express the force in local frame
        F_C2robot_environment_local = mtimes(R02.T,F_RR)
        #Change sign in order to have the force that the enviorment excert to the robot
        F_C2 = - F_C2robot_environment_local
        C2 = mtimes(A2,F_C2)
        
        #Append the constraints
        g.append(vertcat(C1,C2))
        lbg += np.full((1,10),-np.inf)[0].tolist()  
        ubg += np.zeros(10).tolist()
        
        #Force static equilibrium
        g.append(vertcat( F_LR + F_RR + Fdes))
        lbg += np.zeros(3).tolist()  
        ubg += np.zeros(3).tolist()
        
        #Moment static equilibrium
        g.append(vertcat(cross(pL - pR,F_LR) + cross(pR - pL,F_RR)))
        lbg += np.zeros(3).tolist()  
        ubg += np.zeros(3).tolist()
    
        # RELATIVE POSE CONSTRAINT
            
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

        #I have defined the force that the robot exerts on the ambient.
        #The force that the ambient exerts on the robot is equal with a minus sign
        #You need to take care of that while computing the inverse dynamics
       
        #Compute torque at each step
        JLA = Centauro_features.jac_la(qc_k)
        JRA = Centauro_features.jac_ra(qc_k)
        tau = InvDyn(qc_k, np.zeros(nq).tolist(), np.zeros(nq).tolist()) + mtimes(JLA.T,W_LA) + mtimes(JRA.T,W_RA)
#            
        g.append(tau)
        lbg += [lbtorque]
        ubg += [ubtorque]
        lbtq.append(lbtorque)
        ubtq.append(ubtorque)
                
        Tref = np.full((1,nq),50.0)[0].tolist()      
        ###################################################   
        ##                 COST FUNCTION                 ##   
        ###################################################
        J += 1000*dot(pbox - Box_ini , pbox - Box_ini)
        J += 10*dot(qcd_k,qcd_k)
#        J += 0.001*dot(F_LR,F_LR)
#        J += 0.001*dot(F_RR,F_RR)
#        for indexT in range(0,nq):
#            if T_0[indexT] > 0.8 * mpc.temperature_bound:
#                J += 100000 * T_k[indexT]
        #J += 0.001*dot(T_k - Tref, T_k - Tref)
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
        T_next =  np.e**(-T/N/Ttheta) * T_k + Ploss * Rtheta * (1 - np.e**(-T/N/Ttheta))
        
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

    #Allocate a solver
    Solver = nlpsol("solver", mpc.MySolver, nlp, opts)

    if s >= 1:
        r = Solver(x0 = sol,lbx = lbw, ubx = ubw, lbg = lbg, ubg = ubg)  
    else:
        r = Solver(x0 = sol0,lbx = lbw, ubx = ubw, lbg = lbg, ubg = ubg)  
        
#        Solver = nlpsol('Solver','sqpmethod',nlp)
#        r = Solver(lbx = lbw, ubx = ubw, lbg = lbg, ubg = ubg)  
    sol = r['x'].full().flatten()
    soltot.append(sol)
    #Initial condition for the next OCP
    qc_0 = sol[N*n:N*n+nq]
    T_0 = sol[N*n+nq:N*n+2*nq]
    qc_dot0 = sol[k*n+2*nq:k*n+3*nq] 
    
    for t_index in range(nq):
        if T_0[t_index] / mpc.temperature_bound > mpc.Tperc :
            T_0[t_index] =  mpc.Tperc * T_0[t_index]        
    

    RelativePosition_0 = np.round(RelativePosition(qc_0),3)
    RelativeOrientation_0 = np.round(RelativeOrientationError(qc_0),3)

    qc_0 = np.round(qc_0,4).tolist()
    T_0 = np.round(T_0,4).tolist()
    qc_dot0 = np.round(qc_dot0,4).tolist()
   
    #Initialilize the relative position list
    RelPosition = [RelativePosition_0]
    
    print("")
    print("#####################################")
    print("#   Estimated joint temperatures    #")
    print("#####################################")
    print(T_0)
    print("#####################################")
    print("#   End effector relative position  #")
    print("#####################################")
    print(RelativePosition_0) 
    print("#####################################")
    print("# End effector relative orientation #")
    print("#####################################")
    print(RelativeOrientation_0)
    
    data_to_send = Float32MultiArray()
    j = 0
    
    while (j < N):
        #Extract the optimal GLOBAL FORCE
        Ftot_global = sol[j*n+3*nq:j*n+n].tolist()
        #Transform FL  in LOCAL FORCE
        RfromWtoL = np.array(ForwKinLA(sol[j*n:j*n+nq],"rot").T)
        F_tot_opt_local = [ mtimes(RfromWtoL,Ftot_global[0:3]) ]
        
        #Transform FR  in LOCAL FORCE
        RfromWtoR = np.array(ForwKinRA(sol[j*n:j*n+nq],"rot").T)
        F_tot_opt_local.append( mtimes(RfromWtoR,Ftot_global[3:6]))
        
        W_LA = Ftot_global[0:3] + [0.0,0.0,0.0]
        W_RA = Ftot_global[3:6] + [0.0,0.0,0.0]
        J_LA = Centauro_features.jac_la(sol[j*n:j*n+nq])
        J_RA = Centauro_features.jac_ra(sol[j*n:j*n+nq])
        tau = InvDyn(sol[j*n:j*n+nq], np.zeros(nq).tolist(), np.zeros(nq).tolist()) + mtimes(J_LA.T,W_LA) + mtimes(J_RA.T,W_RA)
        
        data_to_send.data =  np.array(sol[j*n:j*n+nq].tolist() + np.concatenate(np.concatenate(F_tot_opt_local).tolist()).tolist() + np.concatenate(np.array(tau).tolist()).tolist() + sol[j*n+nq:j*n+2*nq].tolist())
        pub.publish(data_to_send)
        rate.sleep()
        #Move on the counter
        j += 1 
    
        
    print("OCP has been solved, send trajectory to the unroller ...")
    #Upgrade s to let the warmstarting start and the new OCP
    s = s + 1
    
soltot = np.concatenate(soltot).tolist()
folder = "CsvSolution/"
with open('/home/user/workspace/src/mpc_fatigue/Centauro_solutions/' + folder + '/sol_MPCwithfriction.csv', 'wb') as myfile:
        wr = csv.writer(myfile)
        wr.writerow(soltot)
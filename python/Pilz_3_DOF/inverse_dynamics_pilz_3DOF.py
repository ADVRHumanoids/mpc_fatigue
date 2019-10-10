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
import matplotlib.pyplot as plt
import numpy as np
import talker_inv_dyn_pilz_3DOF as ta
import talker_inv_kin_pilz_3DOF as ti


#----------------------------  Load the urdf  ---------------------------------
urdf = rospy.get_param('/robot_description')
# ---------------------- Solve forward kinematics  ----------------------------
##Postion of link 5 in cartesian space (Solution to direct kinematics)
#fk_string = pin.generate_forward_kin(urdf, 'prbt_link_5')
##Create casadi function
#fk = casadi.Function.deserialize(fk_string)
#
##pos = fk(q = [0.0,-0.3,-0.3])['ee_pos']
#pos = fk(q = [0.785398, 1.28124, 0.0])['ee_pos']
#print('Plotto la posizione')
#print(pos)
#
#        
# ---------------------- Solve Inverse dynamics  ----------------------------
Idyn_string = pin.generate_inv_dyn(urdf)
#Create casadi function
Idyn = casadi.Function.deserialize(Idyn_string)
#print(Idyn)

# ---------------------------   NLP formulation -----------------------------

#Dynamic system
nq = 3
qc = SX.sym('qc', nq)
qcdot = SX.sym('qcdot', nq)
qcddot = np.zeros(nq)

#tau = Idyn(q=qc,qdot= qcdot, qddot = qcddot)['tau']
#print(tau)

#Initial conditions
qc_init = [0.0, 1.2124, -0.5] # High torque on second joint
qcdot_init = np.zeros(nq)

#Set initial condition on Rviz
ti.talker(qc_init)

# Define time 
T = 4.
N = 20
h = T/N
# Define a certain time grid
tgrid = [T/N*k for k in range(N)]

# DEGREES OF FREEDOM
tau0 = 50
alpha = 2
lbqdot = -1.5
ubqdot = 1.5
bound_torque = 10.0

# Empty NLP
w = []
lbw = []
ubw = []
g = []
lbg = []
ubg = []
J = 0
lbt_plot = []
ubt_plot = []

#Initial condition
qc_k = SX.sym('qc0', nq)
qc_k[0] = 0.0
w.append(qc_k[1:3])
lbw += qc_init[1:3]
ubw += qc_init[1:3]

for k in range(N):
    
    #Control at each interval
    qcdname = 'qcd'+ str(k)
    qcd_k = SX.sym(qcdname, nq)
    qcd_k[0] = 0.0
    w.append(qcd_k[1:3])
    if k == 0:
        lbw += qcdot_init[1:3].tolist()
        ubw += qcdot_init[1:3].tolist()
    else:
        lbw += np.full((1,nq),lbqdot)[0][1:3].tolist()
        ubw += np.full((1,nq),ubqdot)[0][1:3].tolist()
    
    #Constraint over tau
    tau = Idyn(q=qc_k, qdot= qcd_k, qddot = qcddot)['tau'][1:3]
    
    g.append(tau)
    esp = alpha * k * h
    torque = tau0 * np.exp(-esp)
    
    if torque > bound_torque:
        ubg += np.full((1,nq), torque)[0][1:3].tolist()   
        lbg += np.full((1,nq), - torque)[0][1:3].tolist() 
        ubt_plot.append(torque)
        lbt_plot.append(- torque)
    else:
        ubg +=  np.full((1,3), bound_torque)[0][1:3].tolist() 
        lbg +=  np.full((1,3),- bound_torque)[0][1:3].tolist() 
        lbt_plot+= [-bound_torque]
        ubt_plot+= [ bound_torque]
            
    #Update J
    J +=  mtimes(tau.T,tau)
    #J += mtimes(qcd_k.T,qcd_k)
    
    #Now integrate
    q_next = qc_k[1:3] + qcd_k[1:3] * h
    
    #New local state
    qname = 'qc' + str(k+1)
    qc_k= SX.sym(qname,nq)
    qc_k[0] = 0.0
    w.append(qc_k[1:3])
    lbw += np.array([-2.53,-2.35]).tolist() #Cambiare ?
    ubw += np.array([2.53,2.35]).tolist()
    
    #Continuity constraint
    g.append(q_next - qc_k[1:3])
    #print(q_next - qc_k)
    lbg += np.zeros(nq)[1:3].tolist()
    ubg += np.zeros(nq)[1:3].tolist()
    

ubg = vertcat(*ubg)
lbg = vertcat(*lbg)
ubw = vertcat(*ubw)
lbw = vertcat(*lbw)
w = vertcat(*w)
g = vertcat(*g)

print("g.shape:",g.shape)
print("lbg.shape:",lbg.shape)
print("ubg.shape:",ubg.shape)
print("x.shape:",w.shape)
print("lbw.shape:",lbw.shape)
print("ubw.shape:",ubw.shape)

# Create the nlp solver
nlp = dict(f = J, g = g, x = w)
ops = {"max_iter" : 100}
Solver = nlpsol('Solver','ipopt', nlp)
r = Solver(lbx = lbw, ubx = ubw, lbg = lbg, ubg = ubg)   
   
sol = r['x'].full().flatten()
qc_opt= []
q1_opt = []
q2_opt = []
q3_opt = []
qcdot_opt = []
q1dot_opt = []
q2dot_opt = []
q3dot_opt = []
# -------------------------- q and qdot vectors ------------------------------
ranghe = np.size(sol)/(nq-1)
for k in range(ranghe):
    if (k % 2 == 0):
        qc_opt.append(sol[2*k:2*k+2])
        q1_opt.append(0.0)
        q2_opt.append(sol[2*k:2*k+2][0])
        q3_opt.append(sol[2*k:2*k+2][1])
    else:
        qcdot_opt.append(sol[2*k:2*k+2])
        q1dot_opt.append(0.0)
        q2dot_opt.append(sol[2*k:2*k+2][0])
        q3dot_opt.append(sol[2*k:2*k+2][1])
# -----------0.00000000e+00,   0.00000000e+00,   0.00000000e+00,----------------- tau vectors ---------------------------------

#Select taus:
tau1_opt = []
tau2_opt = []
tau3_opt = []

#Idyn = casadi.Function.deserialize(Idyn_string)

for k in range(N):
    qc = [q1_opt[k],q2_opt[k],q3_opt[k]]
    qcdot = [q1dot_opt[k],q2dot_opt[k],q3dot_opt[k]]
    tau = Idyn(q=qc,qdot= qcdot, qddot = qcddot)['tau']
    tau1_opt.append(0.0)
    tau2_opt.append(tau[1])
    tau3_opt.append(tau[2])
    
# Plot q1, q2, q3
plt.figure(1)
plt.clf()
tgrid += [T]
plt.plot(tgrid,q1_opt,'-')
plt.plot(tgrid,q2_opt,'-')
plt.plot(tgrid,q3_opt,'-')
plt.legend(['q1_opt','q2_opt','q3_opt'])
plt.title('q variable in time')
plt.grid()


lower = np.full((1,N),lbqdot).tolist()[0]
upper = np.full((1,N),ubqdot).tolist()[0]

plt.figure(2)
plt.clf()
plt.step(tgrid,vertcat(DM.nan(1), q1dot_opt),'-')
plt.step(tgrid,vertcat(DM.nan(1), q2dot_opt),'-')
plt.step(tgrid,vertcat(DM.nan(1), q3dot_opt),'-')
#plt.step(tgrid,vertcat(DM.nan(1), upper),'-')
#plt.step(tgrid,vertcat(DM.nan(1), lower),'-')
plt.legend(['q1dot_opt','q2dot_opt','q3dot_opt','upper','lower'])
plt.title('qdot variable in time')
plt.ylim(lbqdot, ubqdot)
plt.grid()


#PLot torques
plt.figure(3)
plt.clf()
plt.step(tgrid,vertcat(DM.nan(1), ubt_plot),'--')
plt.step(tgrid,vertcat(DM.nan(1), lbt_plot),'--')
plt.step(tgrid,[DM.nan(1)] + tau1_opt,'--')
plt.step(tgrid,[DM.nan(1)] + tau2_opt,'--')
plt.step(tgrid,[DM.nan(1)] + tau3_opt,'--')
plt.legend(['ubg_plot','lgb_plot','tau1_opt','tau2_opt','tau3_opt'])
plt.grid()
plt.title('torque variable in time')


ta.talker(q1_opt,q2_opt,q3_opt)
plt.show()

print('end')
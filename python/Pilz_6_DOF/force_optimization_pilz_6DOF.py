# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 09:14:25 2019

@author: user
"""

# --------------------------- Load libraries ---------------------------------
from casadi import *
import rospy
import mpc_fatigue.pynocchio_casadi as pin
import matplotlib.pyplot as plt
import numpy as np
import talker_inv_dyn_pilz_6DOF as ta
import talker_inv_kin_pilz_6DOF as ti

#----------------------------  Load the urdf  ---------------------------------
urdf = rospy.get_param('/robot_description')

# ---------------------- Solve forward kinematics  ----------------------------
#Postion of link 5 in cartesian space (Solution to direct kinematics)
fk_string = pin.generate_forward_kin(urdf, 'prbt_link_5')
#Create casadi function
fk = casadi.Function.deserialize(fk_string)

# ---------------------- Solve Jacobian of link_5  ----------------------------
#Jacobian for link_5 (Solution to direct kinematics)
jac_string = pin.generate_jacobian(urdf, 'prbt_link_5')
#Create casadi function
jac_dict = casadi.Function.deserialize(jac_string)

# ---------------------- Solve Inverse dynamics  ----------------------------
Idyn_string = pin.generate_inv_dyn(urdf)
#Create casadi function
Idyn = casadi.Function.deserialize(Idyn_string)

# ---------------------------   NLP formulation -----------------------------

#Variables
nq = 6
qc = SX.sym('qc', nq) #joint angles
qcdot = SX.sym('qcdot', nq) # joint velocities
qcddot = np.zeros(nq) # Joint acceleration
x_ini = 0.1
y_ini = 0.4
z_ini = 0.2
theta1_ini = 0.0
theta2_ini = 0.0
theta3_ini = 0.0

# ---------------------- Solve forward kinematics  ----------------------------
#From the desider initial condition, compute the inverse kinematic to determine
#the initial condition

pos_link_5 = fk(q = qc)['ee_pos']
pos_des_link_5 = SX([x_ini,y_ini,z_ini])

des = dot(pos_link_5 - pos_des_link_5, pos_link_5 - pos_des_link_5)
#Nlp problem
prob = dict(x = qc, f = des)
#Create solver interface
solver = nlpsol('solver','ipopt', prob)
sol = solver(x0 = [0.0,0.0,0.0,0.0,0.0,0.0])

# Jacobian value
#jac_end_effector = jac_dict(q = qc)["J"][0:4]
# Position end effector
# pos_end_effector = fk(q=qc)['ee_pos']
# tau with end effector force
#tau = Idyn(q=qc_k, qdot= qcd_k, qddot = qcddot)['tau'] - mtimes(jac_end_effector.T,Force)

#Initial conditions
qc_init = sol['x']
qcdot_init = np.zeros(nq)

ti.talker(qc_init)
# Define time 
T = 2.
N = 60
h = T/N
# Define a certain time grid
tgrid = [T/N*k for k in range(N)]

# DEGREES OF FREEDOM
tau0 = 50
alpha = 2
lbqdot =  -.4
ubqdot =  .4
bound_torque = 15.0

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
w.append(qc_k)
lbw +=  [qc_init] # .tolist()
ubw +=  [qc_init] # .tolist() 

for k in range(N):
    
    #Control at each interval
    qcdname = 'qcd'+ str(k)
    qcd_k = SX.sym(qcdname, nq)
    w.append(qcd_k)
    if k == 0:
        lbw +=  qcdot_init.tolist() 
        ubw +=  qcdot_init.tolist() 
    else:
        lbw +=  np.full((1,nq),lbqdot)[0].tolist() 
        ubw +=  np.full((1,nq),ubqdot)[0].tolist() 
    
    
    #Force at each interval
    F_name = 'Fx'+ str(k)
    Force = SX.sym(F_name,1) # Force at the end effector
    w.append(Force)
    lbw +=  [-np.inf] 
    ubw +=  [ np.inf] 
    
    #Constraint over tau
    jac_end_effector = jac_dict(q = qc_k)["J"][0:6,0:6]
    F = SX(nq,1)
    F[0] = Force
    
    tau = Idyn(q=qc_k, qdot= qcd_k, qddot = qcddot)['tau'] - mtimes(jac_end_effector.T,F)
    g.append(tau)
    esp = alpha * k * h
    torque = tau0 * np.exp(-esp)
    
    if torque > bound_torque:
        ubg += np.full((1,nq), tau0 * np.exp(-esp) )[0].tolist()
        ubt_plot += [tau0 * np.exp(-esp)]
        lbg +=  np.full((1,nq),- tau0 * np.exp(-esp) )[0].tolist()
        lbt_plot += [- tau0 * np.exp(-esp)]
    else:
        ubg +=  np.full((1,nq),bound_torque)[0].tolist() 
        lbg +=  np.full((1,nq),- bound_torque)[0].tolist() 
        lbt_plot+= [-bound_torque]
        ubt_plot+= [ bound_torque]
  
    #Line constraint
    n_fix = 2     
    pos_x_y = fk(q = qc_k)['ee_pos'][0:n_fix]
    ref = [x_ini,y_ini]
    g.append(pos_x_y - ref)
    lbg +=  np.zeros(n_fix).tolist() 
    ubg +=  np.zeros(n_fix).tolist() 
        
    
    #Now integrate
    q_next = qc_k + qcd_k * h
    
    #New local state
    qname = 'qc' + str(k+1)
    qc_k= SX.sym(qname,nq)
    w.append(qc_k)
    lbw += np.full((1,nq), - np.inf )[0].tolist() #Cambiare ?
    ubw += np.full((1,nq), np.inf)[0].tolist() 
    
    #Continuity constraint
    g.append(q_next - qc_k)
    lbg +=  np.zeros(nq).tolist() 
    ubg +=  np.zeros(nq).tolist() 
    
    #Update J
    #J += - mtimes(tau.T,tau)
    #J += -Force
    J +=  - mtimes(F.T,F)
    
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

qc_opt= []
q1_opt = []
q2_opt = []
q3_opt = []
q4_opt = []
q5_opt = []
q6_opt = []
qcdot_opt = []
q1dot_opt = []
q2dot_opt = []
q3dot_opt = []
q4dot_opt = []
q5dot_opt = []
q6dot_opt = []
Fx_opt = []

ranghe = np.size(sol)/(2*nq+1)


        
for k in range(2*ranghe):
    if (k % 2 == 0):
        qc_opt.append(sol[6*k+(k-k/2):6*k+(k-k/2)+nq])
        q1_opt.append(sol[6*k+(k-k/2):6*k+(k-k/2)+nq][0])
        q2_opt.append(sol[6*k+(k-k/2):6*k+(k-k/2)+nq][1])
        q3_opt.append(sol[6*k+(k-k/2):6*k+(k-k/2)+nq][2])
        q4_opt.append(sol[6*k+(k-k/2):6*k+(k-k/2)+nq][3])
        q5_opt.append(sol[6*k+(k-k/2):6*k+(k-k/2)+nq][4])
        q6_opt.append(sol[6*k+(k-k/2):6*k+(k-k/2)+nq][5])
        qcdot_opt.append(sol[6*k+(k-k/2)+nq:6*k+(k-k/2)+2*nq])
        q1dot_opt.append(sol[6*k+(k-k/2)+nq:6*k+(k-k/2)+2*nq][0])
        q2dot_opt.append(sol[6*k+(k-k/2)+nq:6*k+(k-k/2)+2*nq][1])
        q3dot_opt.append(sol[6*k+(k-k/2)+nq:6*k+(k-k/2)+2*nq][2])
        q4dot_opt.append(sol[6*k+(k-k/2)+nq:6*k+(k-k/2)+2*nq][3])
        q5dot_opt.append(sol[6*k+(k-k/2)+nq:6*k+(k-k/2)+2*nq][4])
        q6dot_opt.append(sol[6*k+(k-k/2)+nq:6*k+(k-k/2)+2*nq][5])
        Fx_opt.append(sol[6*k+(k-k/2)+2*nq])
    if (k == 2*N-1):
        k+=1
        qc_opt.append(sol[6*k+(k-k/2):6*k+(k-k/2)+nq])
        q1_opt.append(sol[6*k+(k-k/2):6*k+(k-k/2)+nq][0])
        q2_opt.append(sol[6*k+(k-k/2):6*k+(k-k/2)+nq][1])
        q3_opt.append(sol[6*k+(k-k/2):6*k+(k-k/2)+nq][2])
        q4_opt.append(sol[6*k+(k-k/2):6*k+(k-k/2)+nq][3])
        q5_opt.append(sol[6*k+(k-k/2):6*k+(k-k/2)+nq][4])
        q6_opt.append(sol[6*k+(k-k/2):6*k+(k-k/2)+nq][5])
        
        
#Select taus:
tau1_opt = []
tau2_opt = []
tau3_opt = []
tau4_opt = []
tau5_opt = []
tau6_opt = []

Idyn = casadi.Function.deserialize(Idyn_string)

for k in range(N):
    
    qc = [q1_opt[k],q2_opt[k],q3_opt[k],q4_opt[k],q5_opt[k],q6_opt[k]]
    qcdot = [q1dot_opt[k],q2dot_opt[k],q3dot_opt[k],q4dot_opt[k],q5dot_opt[k],q6dot_opt[k]]
    jac_end_effector = jac_dict(q = qc)["J"][0:6,0:6]
    
    Fend = np.full((nq,1), 0.0)
    Fend[0] = Fx_opt[k]
    
    tau = Idyn(q=qc,qdot= qcdot, qddot = qcddot)['tau'] - mtimes(jac_end_effector.T,Fend)
    tau1_opt.append(tau[0])
    tau2_opt.append(tau[1])
    tau3_opt.append(tau[2])
    tau4_opt.append(tau[3])
    tau5_opt.append(tau[4])
    tau6_opt.append(tau[5])



ta.talker(q1_opt,q2_opt,q3_opt,q4_opt,q5_opt,q6_opt)
#
# ------------------------------ PLOTS --------------------------------------
tgrid.append(T)

# Plot q1, q2, q3
plt.figure(1)
plt.clf()
plt.plot(tgrid,q1_opt,'-')
plt.plot(tgrid,q2_opt,'-')
plt.plot(tgrid,q3_opt,'-')
plt.plot(tgrid,q4_opt,'-')
plt.plot(tgrid,q5_opt,'-')
plt.plot(tgrid,q6_opt,'-')
plt.legend(['q1_opt','q2_opt','q3_opt','q4_opt','q5_opt','q6_opt'])
plt.title('q variable in time plot')
plt.grid()



plt.figure(2)
plt.clf()
plt.step(tgrid,vertcat(DM.nan(1), q1dot_opt),'-')
plt.step(tgrid,vertcat(DM.nan(1), q2dot_opt),'-')
plt.step(tgrid,vertcat(DM.nan(1), q3dot_opt),'-')
plt.step(tgrid,vertcat(DM.nan(1), q4dot_opt),'-')
plt.step(tgrid,vertcat(DM.nan(1), q5dot_opt),'-')
plt.step(tgrid,vertcat(DM.nan(1), q6dot_opt),'-')
plt.legend(['q1dot_opt','q2dot_opt','q3dot_opt','q4dot_opt','q5dot_opt','q6dot_opt'])
plt.title('qdot variable in time plot')
plt.grid()

#
#

# Plot torque
plt.figure(3)
plt.clf()
plt.step(tgrid,vertcat(DM.nan(1), ubt_plot),'--')
plt.step(tgrid,vertcat(DM.nan(1), lbt_plot),'--')
plt.step(tgrid,[DM.nan(1)] + tau1_opt,'--')
plt.step(tgrid,[DM.nan(1)] + tau2_opt,'--')
plt.step(tgrid,[DM.nan(1)] + tau3_opt,'--')
plt.step(tgrid,[DM.nan(1)] + tau4_opt,'--')
plt.step(tgrid,[DM.nan(1)] + tau5_opt,'--')
plt.step(tgrid,[DM.nan(1)] + tau6_opt,'--')
plt.legend(['ubg_plot','lgb_plot','tau1_opt','tau2_opt','tau3_opt','tau4_opt','tau5_opt','tau6_opt'])
plt.title('torque variable in time plot')
plt.grid()


#Plot Fx force
plt.figure(4)
plt.clf()
plt.step(tgrid,vertcat(DM.nan(1), Fx_opt),'-')
plt.legend('Fx_opt')
plt.title('Fx in time plot')
plt.grid()
plt.show()

print('end')

































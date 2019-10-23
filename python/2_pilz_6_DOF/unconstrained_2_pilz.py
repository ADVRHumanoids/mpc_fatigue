# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 14:23:33 2019

@author: user
"""

#FLAGS
plot = 1
# --------------------------- Load libraries ---------------------------------
from casadi import *
import rospy
import mpc_fatigue.pynocchio_casadi as pin
import matplotlib.pyplot as plt
import numpy as np
import two_pilz_talker_inv_dyn_6DOF as ta
import two_pilz_talker_inv_kin_6DOF as ti


#----------------------------  Load the urdf  ---------------------------------
pilz_first = rospy.get_param('/robot_description_1')
pilz_second = rospy.get_param('/robot_description_2')


# ---------------------- Solve forward kinematics pilz 1  ---------------------
#Postion of link 5 in cartesian space (Solution to direct kinematics)
fk_first_string = pin.generate_forward_kin(pilz_first, 'prbt_link_5')
#Create casadi function
fk_first = casadi.Function.deserialize(fk_first_string)

# ---------------------- Solve forward kinematics pilz 2  ---------------------
#Postion of link 5 in cartesian space (Solution to direct kinematics)
fk_second_string = pin.generate_forward_kin(pilz_second, 'prbt_link_5')
#Create casadi function
fk_second = casadi.Function.deserialize(fk_second_string)

# ---------------------- Solve Jacobian of link_5  ----------------------------
#Jacobian for link_5 (Solution to direct kinematics)
jac_first_string = pin.generate_jacobian(pilz_first, 'prbt_link_5')
jac_second_string = pin.generate_jacobian(pilz_second, 'prbt_link_5')
#Create casadi function
jac_first = casadi.Function.deserialize(jac_first_string)
jac_second = casadi.Function.deserialize(jac_second_string)
# ---------------------- Solve Inverse dynamics Pilz 1 and 2  -----------------
Idyn_first_string = pin.generate_inv_dyn(pilz_first)
Idyn_second_string = pin.generate_inv_dyn(pilz_second)
#Create casadi function
Idyn_first = casadi.Function.deserialize(Idyn_first_string)
Idyn_second = casadi.Function.deserialize(Idyn_second_string)

# ---------------------------   NLP formulation -------------------------------

#Variables
nq = 6
# First arm variables
qc_f = SX.sym('qc_f', nq) #joint angles
# Second arm variables
qc_s = SX.sym('qc_s', nq) #joint angles

#Same zero acceleration and velocity for both
qc_dot = np.zeros(nq) # Joint acceleration
qcddot = np.zeros(nq) # Joint acceleration

#Initial position end effector 1
x_ini_first = 0.3
y_ini_first = 0.4
z_ini_first = 0.0
p_init_first = np.array([x_ini_first,y_ini_first,z_ini_first])

#Initial posizione end effector 2
x_ini_second = 0.7
y_ini_second = y_ini_first
z_ini_second = z_ini_first
p_init_second = np.array([x_ini_second,y_ini_second,z_ini_second])

#Initial posizion box
x_ini_box = (x_ini_first + x_ini_second)/2
y_ini_box = y_ini_first
z_ini_box = z_ini_first

# ---------------------- Solve forward kinematics pilz_1 ----------------------------

#From the desider initial condition, compute the inverse kinematic to determine
pos_link_5_first = fk_first(q = qc_f)['ee_pos']
pos_des_link_5 = SX([x_ini_first,y_ini_first,z_ini_first])

des = dot(pos_link_5_first - pos_des_link_5, pos_link_5_first - pos_des_link_5)
#Nlp problem
prob_first = dict(x = qc_f, f = des)
#Create solver interface
solver_first = nlpsol('solver','ipopt', prob_first)
sol_first = solver_first(x0 = [0.0,0.0,0.0,0.0,0.0,0.0])

# ---------------------- Solve forward kinematics pilz_2 ----------------------------
#From the desider initial condition, compute the inverse kinematic to determine
pos_link_5_second = fk_second(q = qc_s)['ee_pos']
pos_des_link_5 = SX([x_ini_second,y_ini_second,z_ini_second])
des = dot(pos_link_5_second - pos_des_link_5, pos_link_5_second - pos_des_link_5)
#Nlp problem
prob_second = dict(x = qc_s, f = des)
#Create solver interface
solver_second = nlpsol('solver','ipopt', prob_second)
sol_second = solver_second(x0 = [0.0,0.0,0.0,0.0,0.0,0.0])

#Initial position pilz 1 and pilz 2
qc_f_init = sol_first['x']
qc_s_init = sol_second['x']
#Visualize initial condition
ti.talker(qc_f_init,qc_s_init)

# ------------------------------- OCP PROBLEM --------------------------------#

# Define time 
T = 2.
N = 120
h = T/N
# Define a certain time grid
tgrid = [T/N*k for k in range(N)]


# DEGREES OF FREEDOM
lbqdot =  -3.0 #velocity lower bound
ubqdot =  3.0  #velocity upper bound
m = 80; #box mass
pos_toll = 0.005 #tollerance in distance between the two arm
# Joint limits taken from urdf
lj = np.array([-2.96,-2.53,-2.35,-2.96,-2.96,-3.12,-2.96,-2.53,-2.35,-2.96,-2.96,-3.12])
uj = np.array([2.96,2.53,2.35,2.96,2.96,3.12,2.96,2.53,2.35,2.96,2.96,3.12])

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

#Robots nitial condition
qc_init = vertcat(qc_f_init,qc_s_init)
qcdot_init = np.zeros(2*nq).tolist()
qcddot = np.zeros(2*nq).tolist()

#Box initial condition
q_box_init  = [x_ini_box, y_ini_box,z_ini_box]

#Initial distance of the two arm I want to preserve
L = dot(p_init_first - p_init_second, p_init_first - p_init_second)


############################# FILL THE NLP IN ################################
#6 states each robot
qc_k = SX.sym('qc0', 2*nq)
w.append(qc_k)
lbw +=  [qc_init] # .tolist()
ubw +=  [qc_init] # .tolist() 

# position of the box in space
p_box = SX.sym('pbox', 3)
w.append(p_box)
lbw +=  q_box_init #.tolist()
ubw +=  q_box_init #.tolist() 

for k in range(N):
    #Control at each interval
    qcdname = 'qcd'+ str(k)
    qcd_k = SX.sym(qcdname, 2*nq)
    w.append(qcd_k)
    if k == 0:
        lbw +=  qcdot_init
        ubw +=  qcdot_init
    elif k == N-1:
        lbw +=  qcdot_init
        ubw +=  qcdot_init
    else:
        lbw +=  np.full((1,2*nq),lbqdot)[0].tolist() 
        ubw +=  np.full((1,2*nq),ubqdot)[0].tolist() 
    
    
    #Force at the end effectors at each interval
    F_name = 'Fz'+ str(k)
    Force = SX.sym(F_name,2) # Force at the pilz 1 end effector 
    w.append(Force)
    lbw +=  np.full((1,2),-np.inf)[0].tolist() 
    ubw +=  np.full((1,2),np.inf)[0].tolist() 
    
    #Force at the first end effector
    F_first = SX(nq,1)
    F_first[2] = Force[0]

    #Force at the second end effector
    F_second = SX(nq,1)
    F_second[2] = Force[1]
    
    #Constraint over the distance between the two end effectors and box
    pos_first = fk_first(q = qc_k[0:6])['ee_pos']
    pos_sec = fk_second(q = qc_k[6:12])['ee_pos']
    g.append( dot(pos_sec - pos_first , pos_sec - pos_first) - L)
    lbg +=  [-pos_toll] #np.zeros(1).tolist()  
    ubg +=  [pos_toll] #np.zeros(1).tolist() 
    
    
    #Constraint movement on z axis first robot (REMOVE COST FUNCTION ON OTHER AXIS)   
    n_fix = 2     
    pos_x_y_first = fk_first(q = qc_k[0:6])['ee_pos'][0:n_fix]
    ref = [x_ini_first,y_ini_first]
    g.append(pos_x_y_first - ref)
    lbg +=  np.full((1,n_fix),-pos_toll)[0].tolist() 
    ubg +=  np.full((1,n_fix),pos_toll)[0].tolist() 
    
    #Constraint movement on z axis second robot (REMOVE COST FUNCTION ON OTHER AXIS)
    pos_x_y_second = fk_second(q = qc_k[6:12])['ee_pos'][0:n_fix]
    ref = [x_ini_second,y_ini_second]
    g.append(pos_x_y_second - ref)
    lbg +=  np.full((1,n_fix),-pos_toll)[0].tolist() 
    ubg +=  np.full((1,n_fix),pos_toll)[0].tolist() 
  
    #Equilibrium equation along z axis
    #Define weight of the box
    Fdes = SX(nq,1)
    Fdes[2] = m * 9.81
    g.append( dot(F_first + F_second - Fdes, F_first + F_second - Fdes))
    lbg +=  [-pos_toll] #np.zeros(1).tolist()  
    ubg +=  [pos_toll] #np.zeros(1).tolist() 
    
    #The p_box x is constrained to be between the two arm
    g.append( p_box - (fk_second(q = qc_k[6:12])['ee_pos'] + fk_first(q = qc_k[0:6])['ee_pos'])/2 )
    lbg +=  np.zeros(3).tolist() 
    ubg +=  np.zeros(3).tolist() 
    
    #Update cost function
#    J += 100*dot( p_box[0] - SX(0.4),p_box[0] - SX(0.4) )
#    J += 100*dot( p_box[1] - SX(0.4),p_box[1] - SX(0.4) )
    J += 100* dot( p_box[2] - SX(0.6),p_box[2] - SX(0.6) )
    
    #Integrate
    q_next = qc_k + qcd_k * h
    
    #New local state
    qname = 'qc' + str(k+1)
    qc_k= SX.sym(qname,2*nq)
    w.append(qc_k)
    lbw += lj.tolist()
    ubw += uj.tolist()
    
    pname = 'pbox' + str(k+1)
    p_box = SX.sym(pname, 3)
    w.append(p_box)
    lbw +=  np.full((1,3),-np.inf)[0].tolist() 
    ubw +=  np.full((1,3),np.inf)[0].tolist() 
    
    #Continuity constraint
    g.append(q_next - qc_k)
    lbg +=  np.zeros(2*nq).tolist() 
    ubg +=  np.zeros(2*nq).tolist() 
    
    #Update J
    #J += - ( mtimes(tau_first.T,tau_first) +mtimes(tau_second.T,tau_second))
    #J += -Force
#    J +=  - mtimes(F_first.T,F_first)
#    J +=  - mtimes(F_second.T,F_second)
#    g.append(p_box[2] - 0.6)
#    lbg +=  [0.0] #np.zeros(1).tolist()  
#    ubg +=  [0.0] #np.zeros(1).tolist()
    
#    g.append(p_box[2] - 0.6)
#    lbg +=  [0.0] #np.zeros(1).tolist()  
#    ubg +=  [0.0] #np.zeros(1).tolist()
#    Fdes = SX(nq,1)
#    Fdes[2] = m * 9.81
    
    
#    Fdes = SX(nq,1)
#    Fdes[2] = 50
#    J +=  dot(F_first - Fdes, F_first - Fdes)
#    J +=  dot(F_second - Fdes, F_second - Fdes)
    
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

# Plot the results
if (plot == 1):
    qc_opt= []
    q1_opt_f = []
    q2_opt_f = []
    q3_opt_f = []
    q4_opt_f = []
    q5_opt_f = []
    q6_opt_f = []
    qcdot_opt = []
    q1dot_opt_f = []
    q2dot_opt_f = []
    q3dot_opt_f = []
    q4dot_opt_f = []
    q5dot_opt_f = []
    q6dot_opt_f = []
    Fzf_opt = []
    
    
    q1_opt_s = []
    q2_opt_s = []
    q3_opt_s = []
    q4_opt_s = []
    q5_opt_s = []
    q6_opt_s = []
    q1dot_opt_s = []
    q2dot_opt_s = []
    q3dot_opt_s = []
    q4dot_opt_s = []
    q5dot_opt_s = []
    q6dot_opt_s = []
    Fzs_opt = []
    
    x_box = []
    y_box = []
    z_box = []
    
    ranghe = np.size(sol)/(4*nq+5)
    
    for k in range(2*ranghe):
        if (k % 2 == 0):
            #JOINT ANGLES
            qc_opt = []
            qc_opt.append(sol[14*k+(k-k/2):14*k+(k-k/2)+2*nq].tolist())
            qc_opt = qc_opt[0]
            #Extract first robot coordinates
            q1_opt_f.append(qc_opt[0])
            q2_opt_f.append(qc_opt[1])
            q3_opt_f.append(qc_opt[2])
            q4_opt_f.append(qc_opt[3])
            q5_opt_f.append(qc_opt[4])
            q6_opt_f.append(qc_opt[5])
            #Extract second robot coordinates
            q1_opt_s.append(qc_opt[6])
            q2_opt_s.append(qc_opt[7])
            q3_opt_s.append(qc_opt[8])
            q4_opt_s.append(qc_opt[9])
            q5_opt_s.append(qc_opt[10])
            q6_opt_s.append(qc_opt[11])
            
            x_box.append(sol[14*k+(k-k/2)+2*nq])
            y_box.append(sol[14*k+(k-k/2)+2*nq+1])
            z_box.append(sol[14*k+(k-k/2)+2*nq+2])
            
            #JOINT VELOCITIES
            qcdot_opt = []
            qcdot_opt.append(sol[14*k+(k-k/2)+2*nq+3:14*k+(k-k/2)+4*nq+3].tolist())
            qcdot_opt = qcdot_opt[0]
            #Extract first robot velocities
            q1dot_opt_f.append(qcdot_opt[0])
            q2dot_opt_f.append(qcdot_opt[1])
            q3dot_opt_f.append(qcdot_opt[2])
            q4dot_opt_f.append(qcdot_opt[3])
            q5dot_opt_f.append(qcdot_opt[4])
            q6dot_opt_f.append(qcdot_opt[5])
            
            #Extract second robot velocities
            q1dot_opt_s.append(qcdot_opt[6])
            q2dot_opt_s.append(qcdot_opt[7])
            q3dot_opt_s.append(qcdot_opt[8])
            q4dot_opt_s.append(qcdot_opt[9])
            q5dot_opt_s.append(qcdot_opt[10])
            q6dot_opt_s.append(qcdot_opt[11])
            
            Fzf_opt.append(sol[14*k+(k-k/2)+4*nq+3])
            Fzs_opt.append(sol[14*k+(k-k/2)+4*nq+1+3])
            
        if (k == 2*N-1):
            k+=1
            qc_opt = []
            #JOINT ANGLES
            qc_opt.append(sol[14*k+(k-k/2):14*k+(k-k/2)+2*nq].tolist())
            qc_opt = qc_opt[0]
            #Extract first robot coordinates
            q1_opt_f.append(qc_opt[0])
            q2_opt_f.append(qc_opt[1])
            q3_opt_f.append(qc_opt[2])
            q4_opt_f.append(qc_opt[3])
            q5_opt_f.append(qc_opt[4])
            q6_opt_f.append(qc_opt[5])
            #Extract second robot coordinates
            q1_opt_s.append(qc_opt[6])
            q2_opt_s.append(qc_opt[7])
            q3_opt_s.append(qc_opt[8])
            q4_opt_s.append(qc_opt[9])
            q5_opt_s.append(qc_opt[10])
            q6_opt_s.append(qc_opt[11])
            
#            x_box.append(sol[14*k+(k-k/2)+2*nq])
#            y_box.append(sol[14*k+(k-k/2)+2*nq+1])
#            z_box.append(sol[14*k+(k-k/2)+2*nq+2])
            
            
            
    #First robot taus:
    tau1_opt_f = []
    tau2_opt_f = []
    tau3_opt_f = []
    tau4_opt_f = []
    tau5_opt_f = []
    tau6_opt_f = []
    
    #Second robot taus:
    tau1_opt_s = []
    tau2_opt_s = []
    tau3_opt_s = []
    tau4_opt_s = []
    tau5_opt_s = []
    tau6_opt_s = []
    
    #First robot torques
    for k in range(N):
        
        qc = [q1_opt_f[k],q2_opt_f[k],q3_opt_f[k],q4_opt_f[k],q5_opt_f[k],q6_opt_f[k]]
        qcdot = [q1dot_opt_f[k],q2dot_opt_f[k],q3dot_opt_f[k],q4dot_opt_f[k],q5dot_opt_f[k],q6dot_opt_f[k]]
        jac_end_effector = jac_first(q = qc)["J"][0:6,0:6]
        
        Fend = np.full((nq,1), 0.0)
        Fend[2] = Fzf_opt[k]
        
        tau = Idyn_first(q=qc,qdot= qcdot, qddot = qcddot[0:6])['tau'] - mtimes(jac_end_effector.T,Fend)
        tau1_opt_f.append(tau[0])
        tau2_opt_f.append(tau[1])
        tau3_opt_f.append(tau[2])
        tau4_opt_f.append(tau[3])
        tau5_opt_f.append(tau[4])
        tau6_opt_f.append(tau[5])
    
    
    for k in range(N):
        
        qc = [q1_opt_s[k],q2_opt_s[k],q3_opt_s[k],q4_opt_s[k],q5_opt_s[k],q6_opt_s[k]]
        qcdot = [q1dot_opt_s[k],q2dot_opt_s[k],q3dot_opt_s[k],q4dot_opt_s[k],q5dot_opt_s[k],q6dot_opt_s[k]]
        jac_end_effector = jac_second(q = qc)["J"][0:6,0:6]
        
        Fend = np.full((nq,1), 0.0)
        Fend[2] = Fzs_opt[k]
        
        tau = Idyn_first(q=qc,qdot= qcdot, qddot = qcddot[0:6])['tau'] - mtimes(jac_end_effector.T,Fend)
        tau1_opt_s.append(tau[0])
        tau2_opt_s.append(tau[1])
        tau3_opt_s.append(tau[2])
        tau4_opt_s.append(tau[3])
        tau5_opt_s.append(tau[4])
        tau6_opt_s.append(tau[5])
    
    
    #Prepare robots configurations
    qf = np.array([q1_opt_f,q2_opt_f,q3_opt_f,q4_opt_f,q5_opt_f,q6_opt_f])
    qs = np.array([q1_opt_s,q2_opt_s,q3_opt_s,q4_opt_s,q5_opt_s,q6_opt_s])
    
    dist = []
    for i in range(np.size(q1_opt_f)):
        qf_p = [qf[0][i],qf[1][i],qf[2][i],qf[3][i],qf[4][i],qf[5][i]]
        qs_p = [qs[0][i],qs[1][i],qs[2][i],qs[3][i],qs[4][i],qs[5][i]]
        pos_sec = fk_first(q = qf_p)['ee_pos']
        pos_first = fk_second(q = qs_p)['ee_pos']
        dist.append( dot(pos_sec - pos_first , pos_sec - pos_first) )
        
    #Prepare applications points
    F = [Fzf_opt,Fzs_opt,Fdes[2]]
    
    box = [x_box,y_box,z_box]
    ta.talker(qf,qs,F,box)
    
    # ------------------------------ PLOTS --------------------------------------
    tgrid.append(T)
    
    # Plot q1, q2, q3
    plt.figure(1)
    plt.clf()
    plt.plot(tgrid,q1_opt_f,'-')
    plt.plot(tgrid,q2_opt_f,'-')
    plt.plot(tgrid,q3_opt_f,'-')
    plt.plot(tgrid,q4_opt_f,'-')
    plt.plot(tgrid,q5_opt_f,'-')
    plt.plot(tgrid,q6_opt_f,'-')
    plt.legend(['q1_opt_f','q2_opt_f','q3_opt_f','q4_opt_f','q5_opt_f','q6_opt_f'])
    plt.title('qfirst variable in time plot')
    plt.grid()
    
    plt.figure(2)
    plt.clf()
    plt.plot(tgrid,q1_opt_s,'-')
    plt.plot(tgrid,q2_opt_s,'-')
    plt.plot(tgrid,q3_opt_s,'-')
    plt.plot(tgrid,q4_opt_s,'-')
    plt.plot(tgrid,q5_opt_s,'-')
    plt.plot(tgrid,q6_opt_s,'-')
    plt.legend(['q1_opt_f','q2_opt_f','q3_opt_f','q4_opt_f','q5_opt_f','q6_opt_f'])
    plt.title('qsecond variable in time plot')
    plt.grid()
    
    plt.figure(3)
    plt.clf()
    plt.step(tgrid,vertcat(DM.nan(1), q1dot_opt_f),'-')
    plt.step(tgrid,vertcat(DM.nan(1), q2dot_opt_f),'-')
    plt.step(tgrid,vertcat(DM.nan(1), q3dot_opt_f),'-')
    plt.step(tgrid,vertcat(DM.nan(1), q4dot_opt_f),'-')
    plt.step(tgrid,vertcat(DM.nan(1), q5dot_opt_f),'-')
    plt.step(tgrid,vertcat(DM.nan(1), q6dot_opt_f),'-')
    plt.legend(['q1dot_opt_f','q2dot_opt_f','q3dot_opt_f','q4dot_opt_f','q5dot_opt_f','q6dot_opt_f'])
    plt.title('qdot first variable in time plot')
    plt.grid()
    
    plt.figure(4)
    plt.clf()
    plt.step(tgrid,vertcat(DM.nan(1), q1dot_opt_s),'-')
    plt.step(tgrid,vertcat(DM.nan(1), q2dot_opt_s),'-')
    plt.step(tgrid,vertcat(DM.nan(1), q3dot_opt_s),'-')
    plt.step(tgrid,vertcat(DM.nan(1), q4dot_opt_s),'-')
    plt.step(tgrid,vertcat(DM.nan(1), q5dot_opt_s),'-')
    plt.step(tgrid,vertcat(DM.nan(1), q6dot_opt_s),'-')
    plt.legend(['q1dot_opt_s','q2dot_opt_s','q3dot_opt_s','q4dot_opt_s','q5dot_opt_s','q6dot_opt_s'])
    plt.title('qdot second variable in time plot')
    plt.grid()
    
    
    # Plot torque
    plt.figure(5)
    plt.clf()
#    plt.step(tgrid,vertcat(DM.nan(1), ubt_plot),'--')
#    plt.step(tgrid,vertcat(DM.nan(1), lbt_plot),'--')
    plt.step(tgrid,[DM.nan(1)] + tau1_opt_f,'--')
    plt.step(tgrid,[DM.nan(1)] + tau2_opt_f,'--')
    plt.step(tgrid,[DM.nan(1)] + tau3_opt_f,'--')
    plt.step(tgrid,[DM.nan(1)] + tau4_opt_f,'--')
    plt.step(tgrid,[DM.nan(1)] + tau5_opt_f,'--')
    plt.step(tgrid,[DM.nan(1)] + tau6_opt_f,'--')
    plt.legend(['tau1_opt_f','tau2_opt_f','tau3_opt_f','tau4_opt_f','tau5_opt_f','tau6_opt_f'])
    plt.title('torque first variable in time plot')
    plt.grid()
    
    plt.figure(6)
    plt.clf()
#    plt.step(tgrid,vertcat(DM.nan(1), ubt_plot),'--')
#    plt.step(tgrid,vertcat(DM.nan(1), lbt_plot),'--')
    plt.step(tgrid,[DM.nan(1)] + tau1_opt_s,'--')
    plt.step(tgrid,[DM.nan(1)] + tau2_opt_s,'--')
    plt.step(tgrid,[DM.nan(1)] + tau3_opt_s,'--')
    plt.step(tgrid,[DM.nan(1)] + tau4_opt_s,'--')
    plt.step(tgrid,[DM.nan(1)] + tau5_opt_s,'--')
    plt.step(tgrid,[DM.nan(1)] + tau6_opt_s,'--')
    plt.legend(['tau1_opt_s','tau2_opt_s','tau3_opt_s','tau4_opt_s','tau5_opt_s','tau6_opt_s'])
    plt.title('torque second variable in time plot')
    plt.grid()
    
    #Plot Fx force
    plt.figure(7)
    plt.clf()
    plt.step(tgrid,vertcat(DM.nan(1), Fzf_opt),'-')
    plt.legend('Fzf_opt')
    plt.title('Fzf in time plot')
    plt.ylim(0,500)
    plt.grid()
    
    plt.figure(8)
    plt.clf()
    plt.step(tgrid,vertcat(DM.nan(1), Fzs_opt),'-')
    plt.legend('Fzs_opt')
    plt.title('Fzs in time plot')
    plt.ylim(0,500)
    plt.grid()
    
      
    plt.figure(9)
    plt.clf()
    plt.step(tgrid,dist,'-')
    plt.legend('dist')
    plt.title('dist')
    plt.ylim(0,0.7)
    plt.grid()
    
    plt.figure(10)
    plt.clf()
    plt.step(tgrid,vertcat(DM.nan(1), x_box),'-')
    plt.step(tgrid,vertcat(DM.nan(1), y_box),'-')
    plt.step(tgrid,vertcat(DM.nan(1), z_box),'-')
    plt.legend(['xbox','ybox','zbox'])
    plt.title(' Box in space')
    #plt.ylim(0,1.0)
    plt.grid()
    
    
    
    plt.show()
    print('end')
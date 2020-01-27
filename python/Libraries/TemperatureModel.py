# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 11:54:45 2019

@author: Pasquale Buonocore

Temperature model
"""
import numpy as np
import matplotlib.pyplot as plt
#==============================================================================
#                    Brashless motor equivalent circuit 
#==============================================================================

#The power loss of the brushless motor can be obtained:

Ra = 10. # [Ohm]
Rh = 2. #[Ohm]

# Ia is the actual current
# E is the actual rotational velocity of the joint

#==============================================================================
#                Thermal model for the brushless servo motor
#==============================================================================

# Tw is the winding temperature
# Ts is the stator temperature
# Ctheta represents the thermal capacity of the stator
# Rtheta0 represents the thermal transfer across the winding insulation to the iron
# Rtheta1 represents the heat transfer between sta stator and the rotor 
# Rtheta2 represents the heat transfer to the surrounding

Rtheta1 = 300. # [ohm] 
Rtheta2 = 9. # [ohm]
Ctheta = 15. # [F]
#If we ignore Rtheta0 and define
Rtheta = (Rtheta1*Rtheta2)/(Rtheta1+Rtheta2)
Ttheta = Rtheta * Ctheta

T = 120. # [s]
N = 200
Tbound = 70.
# We can compute the temperature at each time step 
tgrid = [T/N*k for k in range(N)]


def CompPloss(Ia,E):
    # First component is due to Joule effect
    Pj = (Ia)**2 * Ra
    # Second component is due to all the other losses function of the motor speed
    Ps = E * E / Rh
    return Pj + Ps
    
    
def TempSimulation(Ic,Tin):
    
    Ploss = CompPloss(Ic,0.0)
    Tw = [Tin]
 
    for i in range(N):
        
        if Tw[i] > Tbound:
            print("### Motor temperature will violate the constraint in" + str(tgrid[i]) + "seconds")
            size = np.size(Tw)
            plt.plot(tgrid[0:size],Tw[0:size],'-r',linewidth = 2)
            plt.plot(tgrid[0:size],np.full((size,1),Tbound),'--b',linewidth = 2)
            plt.ylabel("Temperature [C]",fontweight="bold")
            plt.xlabel("Time [s]",fontweight="bold")
            plt.grid()
            plt.title("TEMPERATURE ESTIMATION",fontweight="bold")
            plt.axis([0,T,20,Tbound+10])
            plt.legend(["Testim","Tbound"],loc='lower right')
            plt.show()
            return [True,[1,1,1,0,0,0,0,0,0,0,0,0,0,0]]
            
        Tw.append(  np.e**(-T/N/Ttheta) * Tw[i] + Ploss * Rtheta * (1 - np.e**(-T/N/Ttheta)) )

    plt.plot(tgrid,Tw[0:N],'-r',linewidth = 2)
    plt.plot(tgrid[0:N],np.full((N,1),Tbound),"--b",linewidth = 2)
    plt.ylabel("Temperature [C]",fontweight="bold")
    plt.xlabel("Time [s]",fontweight="bold")
    plt.grid()
    plt.title("TEMPERATURE ESTIMATION",fontweight="bold")
    plt.axis([0,T,20,Tbound+10])
    plt.legend(["Testim","Tbound"],loc='lower right')
    plt.show()
    return False









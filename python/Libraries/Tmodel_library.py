# -*- coding: utf-8 -*-
from casadi import *
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
ktau = SX([40.0,40.0,40.0,40.0,40.0,30.0,50.0,40.0,30.0,40.0,40.0,40.0,45.0,50.0])

def CompPloss(tau,qdot):
    # First component is due to Joule effect
    Ia = tau/ktau
    Pj = mtimes(Ia.T,Ia) * Ra
    # Second component is due to all the other losses function of the motor speed
    Ps = mtimes(qdot.T,qdot) / Rh
    return Pj + Ps
    
# MPC Fatigue

Folders description:

- Launch: It contains the launch files that loads the robotic arm system in Rviz.
- Plotter: It contains the results as images and video.
- Python: It contains the optimal control python script developed.
- Talkers: It contains python scripts that are used to plot the computed optimal solution in Rviz. 
- Urdf: It contains the urdf file that describes the robotic arm system.

## How to run

Navigate to your bashrc file. 

- cd home
- gedit .bashrc

Copy and paste those line in your bash file:

export PYTHONPATH=/home/user/workspace/src/mpc_fatigue/talkers/pilz_3_DOF:$PYTHONPATH
export PYTHONPATH=/home/user/workspace/src/mpc_fatigue/talkers/pilz_6_DOF:$PYTHONPATH
export PYTHONPATH=/home/user/workspace/src/mpc_fatigue/talkers/2_pilz_6_DOF:$PYTHONPATH

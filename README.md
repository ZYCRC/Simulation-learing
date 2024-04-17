# Simulation-learing
Weekly update for progress
Week1 Feb 23th - Mar 3rd  
Sorry for not recording, busy on course project.  
Ubuntu20.04 installed on laptop  
Week2 Mar 4th -  
Learning on PBD ch3, ch4
ch3 is mainly based on motion modeling.
ch4 in progress
Play with xpbd code (https://github.com/TonyZYT2000/PyPBD). Some confusion why we need to divide to odd and even. A modified version of just horizontal, vertical and diagonal is uploaded. It seems work.
![image](n=30.png)  
Play with grape peeling code. Grape peeling code doubles the surface for one is the real surface and one is fixed boundary. It will first run the dist constrain to simulate the distance then run bounbdary constrain to drag the vertex back due to boundary constrain. The it will calculate the energy on the boundary. When the energy exceed limit, setting it to a very low level to consider it as broken.  
![image](boundary_constrain_high_energy.png)
![image](boundary_constrain_low_energy.png)
Peel something new, build a plane from PV lib and set the grasp point and the control trajectory.
![image](peel_plane.png)
![image](peel_plane_animate.png)

Week3-4 Mar 27th- April  
Learning how to use pytorch to optimize control trajectory. Using gradient decsend to optimize trajectory. First pre-define a control trajectory and run the simulation to get target point position. Then reload the model and optimize the trajecotry from initial position.  
The loss is weird since it will suddenly drop at around 500 iterations and converge quickly but from 1-500 it will stay at nealy same loss. I think it might be trying to break the damage constrain for the 1-500 iterations when the boundary is damaged, it will converge soon.
![image](simulation.png)  
![image](test.png)  
![image](loss.png)  

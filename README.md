![Trajectories and range vs. launch angle](Projects/projectile_sweep.png)
Predicts the ballistic trajectory of objects launched into the air, with and without drag. Change the parameters for desired object, and then play the terminal. Current Assumptions: constant wind, quadratic drag, and exponential air density. The altitude doesn't change much, need a very drastic climb for rho to significantly be effected. Has checks written for bounded angle, and if wind direction is incorrect
For 3D, added a check for planar symmetry and crosswind signs
Added a 3D graph
Fixed the 3D graph, as well as the flight_dynamics function. There was incorrect physics and possible edge case of division via 0
Also added a drag force checker, to make sure the magnitudes are correct

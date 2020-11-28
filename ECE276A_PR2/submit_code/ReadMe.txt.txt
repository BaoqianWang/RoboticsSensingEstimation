UCSD ECE 276A Sensing&Estimation of Robots Winter 2020
Baoqian Wang

Description of the file:

1. load_data.py

   This file is provided by the course. It loads the head angles data and lidar data. 

2. p2_utils.py

   This file is provided by the course. It implements the map correlation and ray tracing algorithm. 

3. auxiliary.py

   This file contains some auxiliary functions for Particle Filter SLAM including generate a data
   structure to represent the map, converting the lidar frame to world frame, filtering the lidar scan
   data that is too close or too far. 

4. ParticleSlam.py
   This file implements the Particle Filter SLAM. The functionalities include prediction, update, resample, mapping, and slam. 
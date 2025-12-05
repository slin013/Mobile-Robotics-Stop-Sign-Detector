# Mobile-Robotics-Stop-Sign-Detector

Mobile Robotics Final Project Proposal Document

Names: Abhishek Narang & Serena Lin
Group 13
EECE 5550 Mobile Robotics Final Project Proposal

Option 2: Reference Papers

ORB Algorithm
https://sites.cc.gatech.edu/classes/AY2024/cs4475_summer/images/ORB_an_efficient_alternative_to_SIFT_or_SURF.pdf 

Feature Detection for Traffic Signs
https://www.researchgate.net/publication/221259639_Towards_Real-Time_Traffic_Sign_Recognition_by_Class-Specific_Discriminative_Features

Optimization Problem Statement: We will use ORB (Oriented Fast and Rotated Brief) to detect distinctive features and attributes of road signs (stop signs in particular) and then use the iterative RANSAC (Random Sample Consensus) algorithm to identify matched keypoints between the referenced image and a provided picture or video. This will allow us to detect and then track the target object through the frames of the video.

Experiments
Pictures some that include stop signs and some that don’t
Moving videos to see if it can track the stop signs in real time

Data Acquisition
We will take pictures of stop signs at different angles and at different times of day around Boston
We will also take pictures without stop signs on the same streets
We will take videos to simulate driving/walking past a stop sign

Quantifying Performance
We will know if our implementation is successful if the stop sign is detected when it is in the picture/video vs if it isn’t detected by drawing a box around the sign and producing another picture/video highlighting the located object

We will produce a piece of code that will take input of photos and videos of stop signs and return whether the sign was detected or not. If it is detected, it will circle it in neon and track it throughout a video.

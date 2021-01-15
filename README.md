## Creation of an "HD MAP" using Road Semantic Segmentation
[![Watch the video](https://github.com/AlejandroSoumah/Maxim_Simulator_SegmentationMapping/blob/main/Screenshot_from_Mapping.mov.png)](https://youtu.be/jOfriBMSS-M)

This is the result of road-segmentation and waypoint creation algorithms applied to Carla-Simulator .

The algorithm works very well creating a realistic map that the ego-vehicle (Maxim) can travel through, in simulation this works fantastically.

This algorithm heavily relies on  road-segmentation - it needs to be of good quality (it can also be edited offline if needed) and in the state-estimation.

In the simulator there is almost no state-estimation error, while in the physical Maxim there is at least 0.5 m/100 m of error.

This is the Segmentation and Mapping applied directly to carla simulator.

### To RUN:
   1. Install Python x3.6
   2. Install Carla-Simulator
   3. Install the following libraries:
        - OpenCV
        - Tensorflow 2.0
        - Numpy
   4.Run Carla-Simulator
   5.Run Manual_Script.py

# Hand Tracker
By Marco Morales

## Equipment used
* Intel RealSense d435
* PincherX 100

## How to run 
First, make sure that the Intel software is running by using the following command.

```
realsense-viewer 
```

Next, run the PincherX 100 launch file which will open RVIZ and power the motors so that the arm can not move and be controlled.

```
roslaunch interbotix_xsarm_control xsarm_control.launch robot_model:=px100
```

Finally, run the python script to start the program.

```
python3 HandTracker.py
```


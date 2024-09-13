## Testing the performance of the zed ros2 wrapper:

### using top:
ros2 node list
top -c -p $(pgrep -d',' -f name_of_command)


### using rqt:
install rqt with 

`sudo apt install ros-humble-rqt*`
sudo apt install ros-${ROS_DISTRO}-rqt-top
to launch, just type `rqt` into the terminal 


### visualizing the image topics
source: https://ros2jsguy.medium.com/4-data-visualization-using-ros-2-rviz2-pub-sub-communications-and-javascript-typescript-6e43cde75029




### looking at the rviz2 display
`ros2 launch zed_display_rviz2 display_zed_cam.launch.py camera_model:=zed2`

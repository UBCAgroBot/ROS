#!/usr/bin/env bash
set -ex

'compiling ros2 package'

git clone --branch ${BRANCH_NAME} --single-branch --no-checkout https://github.com/UBCAgroBot/ROS.git
cd ROS

git sparse-checkout init

echo "/*" > .git/info/sparse-checkout
echo "!/.github/" >> .git/info/sparse-checkout  
echo "!/Container/" >> .git/info/sparse-checkout  
echo "!/Experiments/" >> .git/info/sparse-checkout  
echo "!/Misc Scripts/" >> .git/info/sparse-checkout 

git checkout ${BRANCH_NAME}

cd Workspace
rosdep init
rosdep update
rosdep install	--from-paths src -y --ignore-src
colcon build --packages-select custom_interface
colcon build --packages-select node_test
source /opt/ros/humble/setup.bash
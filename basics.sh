export NAME=trial
export PORT=1234
export ROS_MASTER_URI=http://jz:$PORT/
source /home/imin/ros_candy/devel/setup.zsh
source activate candy
roscore -p $PORT&
sleep 2
rosbag play /data/0828/*.bag /data/0905/*.bag -l --skip-empty=40 -i
# rosbag play /data/*.bag /data/*/*.bag -l --skip-empty=40 -i
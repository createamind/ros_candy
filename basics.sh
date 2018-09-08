export NAME=trial
export PORT=1234
export ROS_MASTER_URI=http://jz:$PORT/
source /home/imin/ros_candy/devel/setup.zsh
source activate candy
roscore -p $PORT&
sleep 2
rosbag play /data/0828/left1.bag -l --skip-empty=40 -i
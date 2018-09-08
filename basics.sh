export NAME=bs32
export PORT=1234
export ROS_MASTER_URI=http://jz:$PORT/
source /home/imin/ros_candy/devel/setup.zsh
source activate candy
roscore -p $PORT&
sleep 2

CUDA_VISIBLE_DEVICES=1 rosrun candy wrapper_candy.py -l -t -n -d&
CUDA_VISIBLE_DEVICES=1 rosbag play /data/0828/left1.bag -l -r 10&

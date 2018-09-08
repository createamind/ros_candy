export NAME=normalbeta
export PORT=10000
export ROS_MASTER_URI=http://jz:$PORT/
export CUDA_VISIBLE_DEVICES=0
source /home/kychen/projects/ros_candy/devel/setup.zsh
source /home/kychen/pyenvs/tensorflow2/bin/activate
roscore -p $PORT&
sleep 2
rosrun candy wrapper_candy.py -l -t -n -d&
rosrun candy trainer_candy.py -u -n $NAME&
rosbag play /data/0828/left1.bag -l -r 10
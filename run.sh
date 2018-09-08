export NAME=finalbigdata
export PORT=10001
export ROS_MASTER_URI=http://jz:$PORT/
export CUDA_VISIBLE_DEVICES=2
source /home/kychen/projects/ros_candy/devel/setup.zsh
source /home/kychen/pyenvs/tensorflow2/bin/activate
roscore -p $PORT&
sleep 2
rosrun candy wrapper_candy.py -l -t -n -d&
rosrun candy trainer_candy.py -u -n $NAME&
rosbag play /data/0828/*.bag /data/0829/*.bag /data/0830/*.bag /data/0905/*.bag -l -r 10 --skip-empty=10
# rosbag play /data/0828/left2.bag -l -r 10 --skip-empty=10
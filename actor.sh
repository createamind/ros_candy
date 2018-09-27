export NAME=trial
export PORT=1234
export ROS_MASTER_URI=http://jz:$PORT/
source /home/imin/ros_candy/devel/setup.zsh
source activate candy

CUDA_VISIBLE_DEVICES=1 rosrun candy actor_candy.py -u -n $NAME

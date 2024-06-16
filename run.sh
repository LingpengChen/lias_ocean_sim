roslaunch lias_ocean_sim world_setup.launch # world
roslaunch lias_ocean_sim launch_rexrov.launch # launch robot
roslaunch uuv_control_cascaded_pid joy_velocity.launch uuv_name:=rexrov model_name:=rexrov joy_id:=0
roslaunch uuv_control_cascaded_pid key_board_velocity.launch uuv_name:=rexrov model_name:=rexrov joy_id:=0
# /home/clp/catkin_ws/src/uuv_simulator/uuv_teleop/scripts/vehicle_keyboard_teleop.py
cd ~/catkin_ws/src/lias_ocean_sim/scripts
rosrun lias_ocean_sim rosbag_record.py

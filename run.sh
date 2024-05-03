roslaunch lias_ocean_sim sonar_in_ocean.launch 
# <arg name="standalone" default="true"/>
 

roslaunch uuv_gazebo_worlds ocean_waves.launch 
roslaunch uuv_gazebo_worlds subsea_bop_panel.launch 
roslaunch lias_ocean_sim launch_rexrov.launch 
roslaunch uuv_control_cascaded_pid joy_velocity.launch uuv_name:=rexrov model_name:=rexrov joy_id:=0
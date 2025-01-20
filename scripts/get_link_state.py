#!/usr/bin/env python

import rospy
from gazebo_msgs.srv import GetLinkState
from gazebo_msgs.msg import LinkState

def get_poses():
    try:
        # 等待服务可用
        rospy.wait_for_service('/gazebo/get_link_state')
        
        # 创建服务客户端
        get_link_state = rospy.ServiceProxy('/gazebo/get_link_state', GetLinkState)
        
        # 获取chess_board的pose
        # chess_link_state = get_link_state(link_name='rexrov::rexrov/cameraright_link', reference_frame='world')
        chess_link_state = get_link_state(link_name='chess_board::board', reference_frame='world')
        chess_pose = chess_link_state.link_state.pose
        
        print("camera_link Pose:")
        print("Position:")
        print("x:", chess_pose.position.x)
        print("y:", chess_pose.position.y)
        print("z:", chess_pose.position.z)
        print("\nOrientation:")
        print("x:", chess_pose.orientation.x)
        print("y:", chess_pose.orientation.y)
        print("z:", chess_pose.orientation.z)
        print("w:", chess_pose.orientation.w)

        # 获取thruster_7的pose
        thruster_link_state = get_link_state(link_name='rexrov::rexrov/cameraright_link_optical', reference_frame='world')
        thruster_pose = thruster_link_state.link_state.pose
        
        print("\ncamera link optical Pose:")
        print("Position:")
        print("x:", thruster_pose.position.x)
        print("y:", thruster_pose.position.y)
        print("z:", thruster_pose.position.z)
        print("\nOrientation:")
        print("x:", thruster_pose.orientation.x)
        print("y:", thruster_pose.orientation.y)
        print("z:", thruster_pose.orientation.z)
        print("w:", thruster_pose.orientation.w)
        
    except rospy.ServiceException as e:
        print("Service call failed: %s" % e)

if __name__ == '__main__':
    # 初始化ROS节点
    rospy.init_node('get_poses', anonymous=True)
    
    # 获取位姿
    get_poses()
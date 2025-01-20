#!/usr/bin/env python

import rospy
from gazebo_msgs.srv import GetWorldProperties
from gazebo_msgs.srv import GetModelProperties
from gazebo_msgs.srv import GetModelState

def get_all_links():
    try:
        # 获取世界中所有模型
        rospy.wait_for_service('/gazebo/get_world_properties')
        get_world_props = rospy.ServiceProxy('/gazebo/get_world_properties', GetWorldProperties)
        world_props = get_world_props()
        
        # 获取模型属性服务
        rospy.wait_for_service('/gazebo/get_model_properties')
        get_model_props = rospy.ServiceProxy('/gazebo/get_model_properties', GetModelProperties)
        
        # 遍历所有模型
        for model_name in world_props.model_names:
            print("\nModel:", model_name)
            
            # 获取该模型的所有link
            model_props = get_model_props(model_name)
            print("Links:")
            for link in model_props.body_names:
                print("- " + link)
            
    except rospy.ServiceException as e:
        print("Service call failed: %s" % e)

if __name__ == '__main__':
    # 初始化ROS节点
    rospy.init_node('get_all_links', anonymous=True)
    
    # 获取所有link
    get_all_links()
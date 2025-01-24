#!/usr/bin/env python
import rospy
from gazebo_msgs.msg import LinkStates
from geometry_msgs.msg import Pose

class RexrovPoseMonitor:
    def __init__(self):
        rospy.init_node('rexrov_pose_monitor')
        
        # 订阅link_states话题
        self.sub = rospy.Subscriber(
            '/gazebo/link_states', 
            LinkStates, 
            self.link_states_callback,
            queue_size=1
        )
        self.last_time = None
        
    def link_states_callback(self, msg):
        try:
            # 找到rexrov::base_link的索引
            # timestamp = msg.header.stamp.to_sec()  # 返回浮点数，单位为秒
            # if not self.last_time:
            #     delta_t = timestamp-self.last_time
            #     rospy.INFO(f'delta_t = {delta_t}')
            # self.last_time = timestamp
            
            link_name = 'rexrov::rexrov/base_link'
            if link_name in msg.name:
                index = msg.name.index(link_name)
                pose = msg.pose[index]
                
                # 打印或处理pose信息
                print("Position: x={:.2f}, y={:.2f}, z={:.2f}".format(
                    pose.position.x, 
                    pose.position.y, 
                    pose.position.z
                ))
                print("Orientation: x={:.2f}, y={:.2f}, z={:.2f}, w={:.2f}".format(
                    pose.orientation.x,
                    pose.orientation.y,
                    pose.orientation.z,
                    pose.orientation.w
                ))
                print(pose)
                
        except ValueError:
            rospy.logerr("Link %s not found in link_states", link_name)
            
    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        monitor = RexrovPoseMonitor()
        monitor.run()
    except rospy.ROSInterruptException:
        pass
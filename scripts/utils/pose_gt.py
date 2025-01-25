#!/usr/bin/env python
import rospy
from nav_msgs.msg import Odometry
from math import atan2, asin, pi

def pose_callback(msg):
    # 从Odometry消息中获取四元数
    w = msg.pose.pose.orientation.w
    x = msg.pose.pose.orientation.x
    y = msg.pose.pose.orientation.y
    z = msg.pose.pose.orientation.z
    
    # 计算roll pitch yaw
    roll = atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = asin(2 * (w * y - z * x))
    yaw = atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    
    # 转换为角度
    roll_deg = roll * 180 / pi
    pitch_deg = pitch * 180 / pi
    yaw_deg = yaw * 180 / pi
    
    print(f"Roll: {roll_deg:.2f}°, Pitch: {pitch_deg:.2f}°, Yaw: {yaw_deg:.2f}°")

def listener():
    rospy.init_node('pose_to_rpy', anonymous=True)
    rospy.Subscriber('/rexrov/pose_gt', Odometry, pose_callback)
    rospy.spin()

if __name__ == '__main__':
    try:
        listener()
    except rospy.ROSInterruptException:
        pass
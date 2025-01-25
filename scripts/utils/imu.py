#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Imu
from math import atan2, asin, pi

def imu_callback(msg):
    # 获取四元数
    w = msg.orientation.w
    x = msg.orientation.x
    y = msg.orientation.y
    z = msg.orientation.z
    
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
    rospy.init_node('imu_to_rpy', anonymous=True)
    rospy.Subscriber('/rexrov/imu', Imu, imu_callback)
    rospy.spin()

if __name__ == '__main__':
    try:
        listener()
    except rospy.ROSInterruptException:
        pass
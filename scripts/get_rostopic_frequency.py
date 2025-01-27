#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image, Imu
from nav_msgs.msg import Odometry
from uuv_sensor_ros_plugins_msgs.msg import DVL
from geometry_msgs.msg import TwistWithCovarianceStamped, PoseStamped
from BESTAnP.msg import SonarData
import time
from cv_bridge import CvBridge, CvBridgeError
import cv2

class FrequencyMonitor:
    def __init__(self):
        rospy.init_node('frequency_monitor', anonymous=True)
        self.last_time = None
        self.frequencies = []
        # self.subscriber = rospy.Subscriber('/image_view_sonar/output', Image, self.callback)
        # self.subscriber = rospy.Subscriber('/blueview_p900_ray/sonar_image', Image, self.callback)
        # self.subscriber = rospy.Subscriber('/rexrov/blueview_p900/sonar_image', Image, self.callback)
        # self.subscriber = rospy.Subscriber('/rexrov/rexrov/cameraright/camera_image', Image, self.callback)
        
        # self.subscriber = rospy.Subscriber('/rexrov/pose_gt', Odometry, self.callback)
        # self.subscriber = rospy.Subscriber('/rexrov/imu', Imu, self.callback)
        self.subscriber = rospy.Subscriber('/rexrov/dvl_twist', TwistWithCovarianceStamped, self.callback)
        # self.subscriber = rospy.Subscriber('/charuco_pose', PoseStamped, self.callback)
        # self.subscriber = rospy.Subscriber('/sim/sonar_data_with_pose', SonarData, self.callback)
        
        # 创建CV Bridge
        self.bridge = CvBridge()
        
        # 创建日志文件,使用时间戳命名
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.log_file = open(f'frequency_log_{timestamp}.txt', 'w')
        self.log_file.write("Time,Frequency(Hz)\n")

    def callback(self, msg: Image):
        current_time = msg.header.stamp.to_sec()
        
        if self.last_time is not None:
            # 计算时间差并转换为频率
            time_diff = current_time - self.last_time
            freq = 1.0 / time_diff
            rospy.loginfo(f"Current frequency: {freq:.2f} Hz, time_diff: {time_diff:.4f} s")
            # rospy.loginfo(f"Data: {2f} Hz, time_diff: {time_diff:.4f} s")
        self.last_time = current_time
        
        # try:
        #     # 将ROS图像消息转换为OpenCV格式
        #     cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        #     print(cv_image.shape)
        #     # 显示图像
        #     cv2.imshow('Image View', cv_image)
        #     cv2.waitKey(1)
            
        # except CvBridgeError as e:
        #     rospy.logerr("CvBridge Error: %s", e)
        # pose = msg.pose.pose
                
        # # 打印或处理pose信息
        # print("Position: x={:.2f}, y={:.2f}, z={:.2f}".format(
        #     pose.position.x, 
        #     pose.position.y, 
        #     pose.position.z
        # ))
        # print("Orientation: x={:.2f}, y={:.2f}, z={:.2f}, w={:.2f}".format(
        #     pose.orientation.x,
        #     pose.orientation.y,
        #     pose.orientation.z,
        #     pose.orientation.w
        # ))

    def shutdown(self):
        self.log_file.close()
        rospy.loginfo("Shutting down frequency monitor...")

if __name__ == '__main__':
    try:
        monitor = FrequencyMonitor()
        rospy.on_shutdown(monitor.shutdown)
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
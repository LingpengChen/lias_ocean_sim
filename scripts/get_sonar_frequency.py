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
from gazebo_msgs.srv import GetLinkState
from geometry_msgs.msg import PoseStamped, Pose
import numpy as np
import tf.transformations
import tf

def pose_to_matrix(pose: Pose) -> np.ndarray:
    quaternion = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
   
    position = [pose.position.x, pose.position.y, pose.position.z]
    # # 从四元数获取旋转矩阵
    # rotation_matrix = tf.transformations.quaternion_matrix(quaternion)
    # # 设置平移部分
    # rotation_matrix[0:3, 3] = position

    # 或者方法2：一步到位
    transform_matrix = tf.transformations.compose_matrix(
        translate=position,
        angles=tf.transformations.euler_from_quaternion(quaternion)
    )
    return transform_matrix

class FrequencyMonitor:
    def __init__(self):
        rospy.init_node('frequency_monitor', anonymous=True)
        self.last_time = None
        self.frequencies = []
        # self.subscriber = rospy.Subscriber('/image_view_sonar/output', Image, self.callback)
        # self.subscriber = rospy.Subscriber('/blueview_p900_ray/sonar_image', Image, self.callback)
        self.subscriber = rospy.Subscriber('/rexrov/blueview_p900/sonar_image', Image, self.callback)
        # self.subscriber = rospy.Subscriber('/rexrov/rexrov/cameraright/camera_image', Image, self.callback)

        rospy.wait_for_service('/gazebo/get_link_state')
        self.get_link_state = rospy.ServiceProxy('/gazebo/get_link_state', GetLinkState)
        chess_board_link_state = self.get_link_state(link_name='chess_board::board', reference_frame='world')
        self.T_w_b = pose_to_matrix( chess_board_link_state.link_state.pose )

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
        
        try:
            # 将ROS图像消息转换为OpenCV格式
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            print(cv_image.shape)
            # 显示图像
            cv2.imshow('Image View', cv_image)
            cv2.waitKey(1)
            
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: %s", e)
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
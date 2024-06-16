#!/usr/bin/python3
import rospy
from sensor_msgs.msg import Image, Imu
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge, CvBridgeError
import cv2

def sonar_callback(sonar_image: Image):
    try:
        cv_sonar_image = bridge.imgmsg_to_cv2(sonar_image, "bgr8")
        cv2.imshow("Sonar Image", cv_sonar_image)
        cv2.waitKey(1)  # 处理 OpenCV GUI 事件
    except CvBridgeError as e:
        rospy.logerr("CvBridge Error: {}".format(e))

def listener():
    rospy.init_node('data_listener', anonymous=True)
    sonar_ord_sub = rospy.Subscriber('/rexrov/blueview_p900/sonar_image', Image, sonar_callback)
    rospy.spin()

if __name__ == '__main__':
    bridge = CvBridge()
    try:
        listener()
    except rospy.ROSInterruptException:
        pass
    finally:
        cv2.destroyAllWindows()  # 确保在程序结束时关闭所有 OpenCV 窗口

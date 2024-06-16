#!/usr/bin/python3
import rospy
import message_filters
from sensor_msgs.msg import Image, Imu
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge, CvBridgeError
import cv2
import os

def get_next_rec_directory():
    items = os.listdir('.')
    rec_dirs = [d for d in items if os.path.isdir(d) and d.startswith('rec') and d[3:].isdigit()]
    rec_numbers = [int(d[3:]) for d in rec_dirs]
    if rec_numbers:
        next_number = max(rec_numbers) + 1
    else:
        next_number = 1
    new_rec_dir = f"rec{next_number}"
    os.mkdir(new_rec_dir)
    print(f"Created new directory: {new_rec_dir}")
    return new_rec_dir

index = 0

def sonar_callback(sonar_image: Image):
    try:
        cv_sonar_image = bridge.imgmsg_to_cv2(sonar_image, "bgr8")
        cv2.imshow("Sonar Image", cv_sonar_image)
        cv2.waitKey(1)  # 处理 OpenCV GUI 事件
    except CvBridgeError as e:
        rospy.logerr("CvBridge Error: {}".format(e))
        
def sonar_pose_callback(sonar_image: Image, pose: Odometry):
    global index
    sonar_timestamp = sonar_image.header.stamp.to_nsec()
    pose_timestamp = pose.header.stamp.to_nsec()

    rospy.loginfo("Received synchronized data")
    rospy.loginfo("Sonar image timestamp: {}".format(sonar_timestamp))
    rospy.loginfo("Pose timestamp: {}".format(pose_timestamp))

    try:
        cv_sonar_image = bridge.imgmsg_to_cv2(sonar_image, "bgr8")
        sonar_filename = os.path.join(sonar_dir, "{}_{}.jpg".format(index, sonar_timestamp))
        cv2.imwrite(sonar_filename, cv_sonar_image)
        rospy.loginfo("Saved image: {}".format(sonar_filename))
        index += 1
    except CvBridgeError as e:
        rospy.logerr("CvBridge Error: {}".format(e))
    
    try:
        pose_info = f"{pose_timestamp}, {pose.pose.pose.position.x}, {pose.pose.pose.position.y}, {pose.pose.pose.position.z}, {pose.pose.pose.orientation.x}, {pose.pose.pose.orientation.y}, {pose.pose.pose.orientation.z}, {pose.pose.pose.orientation.w}\n"
        with open(pose_filename, 'a') as pose_file:
            pose_file.write(pose_info)
    except Exception as e:
        rospy.logerr("Error writing to Pose file: {}".format(e))

def imu_callback(data):
    file_name = new_rec_dir + '/imu_data.txt'
    try:
        imu_timestamp = data.header.stamp.to_nsec()
        imu_info = f"{imu_timestamp}, {data.orientation.x}, {data.orientation.y}, {data.orientation.z}, {data.orientation.w}, {data.angular_velocity.x}, {data.angular_velocity.y}, {data.angular_velocity.z}, {data.linear_acceleration.x}, {data.linear_acceleration.y}, {data.linear_acceleration.z}\n"
        with open(file_name, 'a') as imu_file:
            imu_file.write(imu_info)
    except Exception as e:
        rospy.logerr("Error writing to IMU file: {}".format(e))

def pose_callback(data):
    file_name = new_rec_dir + '/pose_data_continuous.txt'
    try:
        pose_timestamp = data.header.stamp.to_nsec()
        pose_info = f"{pose_timestamp}, {data.pose.pose.position.x}, {data.pose.pose.position.y}, {data.pose.pose.position.z}, {data.pose.pose.orientation.x}, {data.pose.pose.orientation.y}, {data.pose.pose.orientation.z}, {data.pose.pose.orientation.w}\n"
        with open(file_name, 'a') as pose_file:
            pose_file.write(pose_info)
    except Exception as e:
        rospy.logerr("Error writing to Pose file: {}".format(e))

def listener():
    rospy.init_node('data_listener', anonymous=True)

    sonar_ord_sub = rospy.Subscriber('/rexrov/blueview_p900/sonar_image', Image, sonar_callback)
    sonar_sub = message_filters.Subscriber('/rexrov/blueview_p900/sonar_image', Image)
    pose_sub = message_filters.Subscriber('/rexrov/pose_gt', Odometry)
    
    sonar_syn = message_filters.ApproximateTimeSynchronizer([sonar_sub, pose_sub], queue_size=100, slop=0.1)
    sonar_syn.registerCallback(sonar_pose_callback)

    high_freq_pose_sub = rospy.Subscriber('/rexrov/pose_gt', Odometry, pose_callback)
    imu_sub = rospy.Subscriber('/rexrov/imu', Imu, imu_callback)

    rospy.spin()

if __name__ == '__main__':
    new_rec_dir = get_next_rec_directory()
    bridge = CvBridge()
    sonar_dir = './' + new_rec_dir + '/sonar_image'
    pose_filename = './' + new_rec_dir + '/pose_data.txt'
    os.makedirs(sonar_dir, exist_ok=True)

    try:
        listener()
    except rospy.ROSInterruptException:
        pass
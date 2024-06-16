#!/usr/bin/env python
import rospy
import numpy as np
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseArray, Pose, Point, Quaternion

class PoseEstimator:
    def __init__(self, pose_continuous_filename, image_N1, img_match):
        self.pose_continuous_filename = pose_continuous_filename
        self.image_N1 = image_N1
        self.img_match = img_match
        self.trajectory = self.get_trajectory(pose_continuous_filename)
        self.bridge = CvBridge()

        self.image_pub_1 = rospy.Publisher("/original_sonar_frame", Image, queue_size=10)
        self.image_pub_2 = rospy.Publisher("/selected_matches", Image, queue_size=10)
        self.pose_pub = rospy.Publisher("/trajectory", PoseArray, queue_size=10)

    def get_trajectory(self, filename):
        positions = []
        orientations = []
        times = []

        with open(filename, 'r') as file:
            lines = file.readlines()

            for line in lines:
                parts = line.strip().split(', ')
                time = float(parts[0])
                position = [float(parts[1]), float(parts[2]), float(parts[3])]
                orientation = [float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])]

                times.append(time)
                positions.append(position)
                orientations.append(orientation)

        # 平移所有坐标点，使第一个坐标点变为原点 (0, 0, 0)
        positions = np.array(positions)
        positions -= positions[0]

        trajectory = {
            'times': np.array(times),
            'positions': positions,
            'orientations': np.array(orientations)
        }

        return trajectory

    def publish_images(self):
        sonar_img_msg = self.bridge.cv2_to_imgmsg(cv2.cvtColor(self.image_N1, cv2.COLOR_BGR2RGB), "rgb8")
        matches_img_msg = self.bridge.cv2_to_imgmsg(cv2.cvtColor(self.img_match, cv2.COLOR_BGR2RGB), "rgb8")
        
        self.image_pub_1.publish(sonar_img_msg)
        self.image_pub_2.publish(matches_img_msg)

    def publish_trajectory(self):
        pose_array_msg = PoseArray()
        pose_array_msg.header.stamp = rospy.Time.now()
        pose_array_msg.header.frame_id = "map"

        for i in range(len(self.trajectory['positions'])):
            pose = Pose()
            pose.position = Point(*self.trajectory['positions'][i])
            pose.orientation = Quaternion(*self.trajectory['orientations'][i])
            pose_array_msg.poses.append(pose)
        
        self.pose_pub.publish(pose_array_msg)




def main():
    rospy.init_node('pose_estimator')
    pose_estimator = PoseEstimator(
        'pose_data_continuous.txt',  # 替换为实际文件路径
        cv2.imread('./sonar_image/0_1073680000000.jpg'),  # 替换为实际图像路径
        cv2.imread('./sonar_image/1_1076194000000.jpg')  # 替换为实际图像路径
    )

    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        pose_estimator.publish_images()
        pose_estimator.publish_trajectory()
        rate.sleep()

if __name__ == '__main__':
    main()

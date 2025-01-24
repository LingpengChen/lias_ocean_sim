#!/usr/bin/env python3
import cv2
import numpy as np
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from queue import Queue
import threading
from geometry_msgs.msg import PoseStamped, Pose
import tf.transformations as tf_trans
import cv2.aruco as aruco
from gazebo_msgs.srv import GetLinkState
import tf.transformations
import tf

import numpy as np
from scipy.spatial.transform import Rotation


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

def quaternion_to_euler(quaternion):
    # 创建tf变换对象
    explicit_quat = [quaternion.x, quaternion.y, quaternion.z, quaternion.w]
    euler = tf.transformations.euler_from_quaternion(explicit_quat)
    roll = euler[0]
    pitch = euler[1]
    yaw = euler[2]
    return roll, pitch, yaw

def plot_sonar(marking_pts_w, sonar_pos_w, img_width=1200, img_height=600, fov_horizontal=np.pi/3, range_max=5.0):
    # sonar_pos_w是4x4的变换矩阵，我们需要求其逆来得到相对位置
    sonar_pos_w_inv = np.linalg.inv(sonar_pos_w)
    
    # 计算特征点在声纳坐标系下的坐标
    pts_in_sonar = sonar_pos_w_inv @ marking_pts_w
    
    # 只取前两维计算距离和角度（x和y坐标）
    # 由于是齐次坐标，需要除以最后一维使其归一化
    x = pts_in_sonar[0,:]
    y = pts_in_sonar[1,:] 
    z = pts_in_sonar[2,:]
    
    # 计算距离和角度
    distances = np.sqrt(x**2 + y**2 + z**2)
    angles = np.arctan2(y, x)  # 相对于x轴的角度,弧度制
    
    # 创建白色背景图像
    # 绘制黑色扇形
    image = np.ones((img_height, img_width, 3), dtype=np.uint8) * 255
    radius = int(img_height)
    start_angle = -int(np.rad2deg(fov_horizontal)/2)
    
    center = (int(img_width/2), img_height)  # 改为底部中心点
    end_angle = int(np.rad2deg(fov_horizontal)/2)
    cv2.ellipse(image, center, (radius, radius), -90, start_angle, end_angle, (0, 0, 0), -1)  # 角度改为-90
    
    
    # 绘制特征点
    for d, a in zip(distances, angles):
        if d <= range_max and abs(a) <= fov_horizontal/2:  # 只绘制在视场角和最大距离范围内的点
            x_img = int((d * np.sin(a) / range_max) * (img_width/2) + (img_width/2))
            y_img = int(img_height - (d * np.cos(a) / range_max) * img_height)  # 加上img_height并取反
            # print(x_img, y_img)
            cv2.circle(image, (x_img, y_img), 3, (255, 255, 255), -1)
            # cv2.circle(image, (x_img, y_img), 3, (0, 0, 0), -1)
  
    return image, distances, angles


class Sonar_preprocessor:
    def __init__(self):
        rospy.init_node('sonar_preprocessor', anonymous=True)
        self.last_time = None
        self.img_num = 0
        
        # 初始化队列和线程锁
        self.image_queue = Queue(maxsize=10)
        self.bridge = CvBridge()
        
        # 初始化ROS订阅者和发布者
        self.subscriber = rospy.Subscriber('/rexrov/blueview_p900/sonar_image', Image, self.callback)

        self.pose_publisher = rospy.Publisher('/charuco_pose', 
                                            PoseStamped, 
                                            queue_size=10)
        
        # 等待服务可用
        rospy.wait_for_service('/gazebo/get_link_state')
        # 创建服务客户端
        self.get_link_state = rospy.ServiceProxy('/gazebo/get_link_state', GetLinkState)
        
        chess_board_link_state = self.get_link_state(link_name='chess_board::board', reference_frame='world')
        self.T_w_b = pose_to_matrix( chess_board_link_state.link_state.pose )

        marking_pts_b = np.array([[0.15,0.4,0], [-0.15,0.4,0], 
                                       [0.0 ,0.6,0], 
                                       [0.15,0.8,0], [-0.15,0.8,0],
                                       [0.0 ,1.0,0], 
                                       [0.15,1.2,0], [-0.15,1.2,0],
                                       ]) 
        ones = np.ones((marking_pts_b.shape[0], 1))
        marking_pts_b_aug = np.hstack((marking_pts_b, ones))
        self.marking_pts_w = self.T_w_b @ marking_pts_b_aug.T
        # self.marking_pts_w = (self.T_w_b @ marking_pts_b_aug.T)[0:3, :]
        
        # 启动处理线程
        self.processing_thread = threading.Thread(target=self.preprocess_sonar_image_thread)
        self.processing_thread.daemon = True
        self.processing_thread.start()

    def callback(self, msg: Image):
        """图像订阅回调函数"""
        current_time = msg.header.stamp.to_sec()
        # if self.last_time is not None:
        #     # 计算时间差并转换为频率
        #     time_diff = current_time - self.last_time
        #     freq = 1.0 / time_diff
        #     rospy.loginfo(f"Current frequency: {freq:.2f} Hz, time_diff: {time_diff:.4f} s")
        # self.last_time = current_time
            
        try:
            # self.calculate_gt()
            if self.image_queue.full():
                self.image_queue.get()
                rospy.loginfo("Error in image get: image queue is full!")
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
           
            self.image_queue.put((cv_image, msg.header.stamp, self.img_num))
            self.img_num += 1
        except Exception as e:
            rospy.logerr(f"Error in callback: {str(e)}")

    def calculate_gt(self):
        # 获取thruster_7的pose
        robot_link_state = self.get_link_state(link_name='rexrov::rexrov/base_link', reference_frame='world')  
        # print(f"camera pose: {quaternion_to_euler(robot_link_state.link_state.pose.orientation)}")
        T_w_robot = pose_to_matrix( robot_link_state.link_state.pose )
        T_r_s = np.array([[1,0,0,1.2],
                            [0,1,0,0.5],
                            [0,0,1,-0.65],
                            [0,0,0,1]]) # 1.2 0.5 -0.65
        T_w_sonar = T_r_s @ T_w_robot
        img, distances, angles = plot_sonar(self.marking_pts_w, T_w_sonar)
        # print(distances)
        # print(angles)
        cv2.imshow('Sonar View', img)
        cv2.waitKey(1)
        # print(T_w_robot)
        # print(T_w_sonar)
        # print()
        
        
        # chess_board_link_state = self.get_link_state(link_name='chess_board::board', reference_frame='world')
        # self.T_w_b = pose_to_matrix( chess_board_link_state.link_state.pose )
        # print(f"Board pose: {quaternion_to_euler(chess_board_link_state.link_state.pose.orientation)}")
        
        # 绕y轴逆时针转90度 (π/2)
        R_x = np.array([
            [1, 0, 0],
            [0, 0, -1],
            [0, 1, 0]
        ])
        
        Ry = np.array([
            [0,  0, 1],
            [0,  1, 0],
            [-1, 0, 0]
        ])

        # 绕z轴ni时针转90度 (π/2)
        Rz = np.array([
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, 1]
        ])
        # R_w_c = np.linalg.inv(Rz) @ Ry @ T_w_c[:3, :3]
        # R_w_b = np.linalg.inv(Rz) @ Ry @ Ry @ self.T_w_b[:3, :3]
        # R_gt = R_w_c.T @ R_w_b
        # clp = np.array([
        #     [0, 1, 0],
        #     [1, 0, 0],
        #     [0, 0, 1]
        # ])
        # R_gt = clp @ R_gt @ clp.T
        
        # T_c_b = np.linalg.inv(T_w_c) @ self.T_w_b
        # t_temp = T_c_b[:3, 3]
        # t_gt = np.array([-t_temp[1]-0.25, -t_temp[2]-0.25, t_temp[0]]).reshape(3, 1)
        # r_gt, _ = cv2.Rodrigues(R_gt)    
                

    def preprocess_sonar_image_thread(self):
        """处理图像队列的主循环"""
        while not rospy.is_shutdown():
            if not self.image_queue.empty():
                image, timestamp, img_num = self.image_queue.get()
        
                cv2.imshow('Image View', image)
                cv2.waitKey(1)
                    
           
                
                

    def run(self):
        """运行节点"""
        try:
            rospy.spin()
        except KeyboardInterrupt:
            rospy.loginfo("Shutting down Sonar_preprocessor node")

def main():
    sonar_processor = Sonar_preprocessor()
    sonar_processor.run()

if __name__ == '__main__':
    print("OpenCV version:", cv2.__version__)
    print("Available aruco functions:", [x for x in dir(cv2.aruco) if 'pose' in x.lower()])
    main()
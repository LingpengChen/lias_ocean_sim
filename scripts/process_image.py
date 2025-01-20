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

def draw_pose(img, rvec, tvec, camera_matrix, dist_coeffs, length=0.1):
    """
    绘制坐标轴来可视化位姿
    img: 输入图像
    rvec: 旋转向量
    tvec: 平移向量
    camera_matrix: 相机内参矩阵
    dist_coeffs: 畸变系数
    length: 坐标轴长度（米）
    """
    # 定义坐标轴点
    axis_points = np.float32([[0,0,0], 
                             [length,0,0], 
                             [0,length,0], 
                             [0,0,length]]).reshape(-1,3)
    
    # 将3D点投影到图像平面
    img_points, _ = cv2.projectPoints(axis_points, 
                                    rvec, 
                                    tvec, 
                                    camera_matrix, 
                                    dist_coeffs)
    img_points = img_points.astype(int)
    
    # 绘制坐标轴
    origin = tuple(img_points[0].ravel())
    img = cv2.line(img, origin, tuple(img_points[1].ravel()), (0,0,255), 2)  # X轴 红色
    img = cv2.line(img, origin, tuple(img_points[2].ravel()), (0,255,0), 2)  # Y轴 绿色
    img = cv2.line(img, origin, tuple(img_points[3].ravel()), (255,0,0), 2)  # Z轴 蓝色
    
    return img

# 使用示例
def visualize_charuco_pose(image, rvec, tvec, camera_matrix, dist_coeffs):
    """
    完整的ChArUco板位姿可视化
    """
    # 复制图像以免修改原图
    vis_img = image.copy()
    
    # 绘制坐标轴
    vis_img = draw_pose(vis_img, rvec, tvec, camera_matrix, dist_coeffs)
    
    # 显示位姿信息
    height = vis_img.shape[0]
    # 转换旋转向量为欧拉角(度)
    rot_mat = cv2.Rodrigues(rvec)[0]
    euler_angles = cv2.decomposeProjectionMatrix(np.hstack((rot_mat, tvec)))[6] * np.pi / 180.0
    
    # 在图像上显示欧拉角和平移向量
    text_pos_y = 30
    tvec_str = np.round(tvec.ravel(), decimals=2)  # 保留3位小数
    euler_str = np.round(euler_angles.ravel(), decimals=2)  # 保留2位小数
    cv2.putText(vis_img, f"Rotation (deg): {euler_str}", 
                (10, text_pos_y+30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    cv2.putText(vis_img, f"Translation (m): {tvec_str}", 
                (10, text_pos_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    
    return vis_img


class CharucoPoseEstimator:
    def __init__(self):
        rospy.init_node('charuco_pose_estimator', anonymous=True)
        self.last_time = None
        self.img_num = 0
        
        # 初始化相机参数 
        self.camera_matrix = np.array([
            [772.98, 0, 320],
            [0, 772.98, 240],
            [0, 0, 1]
        ], dtype=np.float32)
        self.dist_coeffs = np.zeros(5)
        
        # 初始化ChArUco板参数
        self.dictionary = aruco.Dictionary_get(aruco.DICT_6X6_250)
        
        # 使用旧版API创建ChArUco板
        self.board = aruco.CharucoBoard_create(
            squaresX=5,
            squaresY=5,
            squareLength=0.1,
            markerLength=0.05,
            dictionary=self.dictionary
        )
        
        # 初始化队列和线程锁
        self.image_queue = Queue(maxsize=10)
        self.bridge = CvBridge()
        
        # 初始化ROS订阅者和发布者
        self.subscriber = rospy.Subscriber('/rexrov/rexrov/cameraright/camera_image', 
                                         Image, 
                                         self.callback)
        self.pose_publisher = rospy.Publisher('/charuco_pose', 
                                            PoseStamped, 
                                            queue_size=10)
        
        # 等待服务可用
        rospy.wait_for_service('/gazebo/get_link_state')
        # 创建服务客户端
        self.get_link_state = rospy.ServiceProxy('/gazebo/get_link_state', GetLinkState)
        
        chess_board_link_state = self.get_link_state(link_name='chess_board::board', reference_frame='world')
        self.T_w_b = pose_to_matrix( chess_board_link_state.link_state.pose )

        # 启动处理线程
        self.processing_thread = threading.Thread(target=self.process_images)
        self.processing_thread.daemon = True
        self.processing_thread.start()

    def callback(self, msg):
        """图像订阅回调函数"""
        current_time = msg.header.stamp.to_sec()
        # if self.last_time is not None:
        #     # 计算时间差并转换为频率
        #     time_diff = current_time - self.last_time
        #     freq = 1.0 / time_diff
        #     # rospy.loginfo(f"Current frequency: {freq:.2f} Hz, time_diff: {time_diff:.4f} s")
        # self.last_time = current_time
            
        try:
            if self.image_queue.full():
                self.image_queue.get()
                rospy.loginfo("Error in image get: image queue is full!")
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.image_queue.put((cv_image, msg.header.stamp, self.img_num))
            self.img_num += 1
        except Exception as e:
            rospy.logerr(f"Error in callback: {str(e)}")

    def estimate_pose(self, image):
        """估计单张图像的位姿"""
        try:
            # 转换为灰度图
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # 1. 检测ArUco标记
            corners, ids, rejected = aruco.detectMarkers(
                image=gray,
                dictionary=self.dictionary
            )

            if ids is None:
                return False, None, None

            # 2. 细化并检测ChArUco角点
            ret, charucoCorners, charucoIds = aruco.interpolateCornersCharuco(
                markerCorners=corners,
                markerIds=ids,
                image=gray,
                board=self.board
            )

            if charucoCorners is None or charucoIds is None:
                return False, None, None

            # 3. 估计位姿
            retval, rvec, tvec = aruco.estimatePoseCharucoBoard(
                charucoCorners=charucoCorners,
                charucoIds=charucoIds,
                board=self.board,
                cameraMatrix=self.camera_matrix,
                distCoeffs=self.dist_coeffs
            )

            if retval:
                return True, rvec, tvec
            return False, None, None
            
        except Exception as e:
            rospy.logerr(f"Error in pose estimation: {str(e)}")
            return False, None, None

    def create_pose_msg(self, rvec, tvec, timestamp):
        """创建PoseStamped消息"""
        pose_msg = PoseStamped()
        pose_msg.header.stamp = timestamp
        pose_msg.header.frame_id = "camera_frame"

        rmat, _ = cv2.Rodrigues(rvec)
        quat = tf_trans.quaternion_from_matrix(
            np.vstack((np.hstack((rmat, tvec)), [0, 0, 0, 1]))
        )

        pose_msg.pose.position.x = tvec[0][0]
        pose_msg.pose.position.y = tvec[1][0]
        pose_msg.pose.position.z = tvec[2][0]
        pose_msg.pose.orientation.x = quat[0]
        pose_msg.pose.orientation.y = quat[1]
        pose_msg.pose.orientation.z = quat[2]
        pose_msg.pose.orientation.w = quat[3]

        return pose_msg

    def process_images(self):
        """处理图像队列的主循环"""
        while not rospy.is_shutdown():
            if not self.image_queue.empty():
                image, timestamp, img_num = self.image_queue.get()
                
                # 获取thruster_7的pose
                camera_optical_link_state = self.get_link_state(link_name='rexrov::rexrov/cameraright_link', reference_frame='world')
                T_w_c = pose_to_matrix( camera_optical_link_state.link_state.pose )
                # print(f"camera pose: {quaternion_to_euler(camera_optical_link_state.link_state.pose.orientation)}")
                
                
                chess_board_link_state = self.get_link_state(link_name='chess_board::board', reference_frame='world')
                self.T_w_b = pose_to_matrix( chess_board_link_state.link_state.pose )
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
                R_w_c = np.linalg.inv(Rz) @ Ry @ T_w_c[:3, :3]
                R_w_b = np.linalg.inv(Rz) @ Ry @ Ry @ self.T_w_b[:3, :3]
                R_gt = R_w_c.T @ R_w_b
                clp = np.array([
                    [0, 1, 0],
                    [1, 0, 0],
                    [0, 0, 1]
                ])
                R_gt = clp @ R_gt @ clp.T
                
                T_c_b = np.linalg.inv(T_w_c) @ self.T_w_b
                t_temp = T_c_b[:3, 3]
                t_gt = np.array([-t_temp[1]-0.25, -t_temp[2]-0.25, t_temp[0]]).reshape(3, 1)
                r_gt, _ = cv2.Rodrigues(R_gt)
                # pose_msg = self.create_pose_msg(r, t, timestamp)
                # self.pose_publisher.publish(pose_msg)
                
                
                success, r_est, t_est = self.estimate_pose(image)
                    

                if success:
                    vis_image = visualize_charuco_pose(image, r_est, t_est, self.camera_matrix, self.dist_coeffs)
                    # 显示结果
                    cv2.imshow('Pose Visualization', vis_image)
                    cv2.waitKey(1)
                    # rospy.loginfo(f"Img_num: {img_num}")
                    t_est +=  np.array([[ 0.044], [-0.052], [-0.024]])
                    pose_msg = self.create_pose_msg(r_est, t_est, timestamp)
                    self.pose_publisher.publish(pose_msg)
                    # rospy.loginfo(f"Published pose at time: {timestamp}")
                    
                    current_time = timestamp.to_sec()
                    if self.last_time is not None:
                        # 计算时间差并转换为频率
                        time_diff = current_time - self.last_time
                        freq = 1.0 / time_diff
                        print("R_error:", np.linalg.norm( r_gt.T - r_est.T), "\nT_error: ", np.linalg.norm(t_gt.T-t_est.T) )
                        rospy.loginfo(f"Current frequency: {freq:.2f} Hz, time_diff: {time_diff:.4f} s ") 
                        # rospy.loginfo(f"  Ground_truth r: {r.T}, Estimated R: {rvec.T}")
                        # rospy.loginfo(f"  Ground_truth t: {np.linalg.norm(t)}, Estimated t: {np.linalg.norm(tvec)}")
                    self.last_time = current_time
            # else:
            #     rospy.sleep(0.01)

    def run(self):
        """运行节点"""
        try:
            rospy.spin()
        except KeyboardInterrupt:
            rospy.loginfo("Shutting down CharucoPoseEstimator node")

def main():
    estimator = CharucoPoseEstimator()
    estimator.run()

if __name__ == '__main__':
    print("OpenCV version:", cv2.__version__)
    print("Available aruco functions:", [x for x in dir(cv2.aruco) if 'pose' in x.lower()])
    main()
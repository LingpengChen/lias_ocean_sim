import cv2
import os
import numpy as np
from tri import sonar_triangulation
from utils import pose_to_transform_matrix
from anp import AnPAlgorithm
import rospy
from cv_bridge import CvBridge

from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseArray, Pose, Point, Quaternion
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped


# left right corner is (0,0) bottom right (50,100)
def convert_points_to_polar(points, width, height): # 512 # 798
    # 参考点(center)
    center = (width / 2, height)

    # 提取x和y坐标
    x_coords = -points[:, 0, 0] + width
    y_coords = points[:, 0, 1]

    # 计算距离(distance)
    distances = 20/height*np.sqrt((x_coords - center[0])**2 + (y_coords - center[1])**2)

    # 计算角度(Theta)（以弧度表示）
    thetas = np.arctan2(center[1] - y_coords, x_coords - center[0]) - np.pi/2

    return thetas, distances

def read_data(filename, init_pose):
    times = []
    poses = []

    with open(filename, 'r') as file:
        lines = file.readlines()

        for line in lines:
            parts = line.strip().split(', ')
            pose_timestamp = int(parts[0])
            pose = {
                'position': {
                    'x': float(parts[1]) - init_pose['position']['x'],
                    'y': float(parts[2]) - init_pose['position']['y'],
                    'z': float(parts[3]) - init_pose['position']['z'],
                },
                'orientation': {
                    'x': float(parts[4]),
                    'y': float(parts[5]),
                    'z': float(parts[6]),
                    'w': float(parts[7]),
                }
            }
            times.append(pose_timestamp)
            poses.append(pose)

    # 平移所有坐标点，使第一个坐标点变为原点 (0, 0, 0)
    # poses = np.array(poses)
    # poses['position'] -= poses['position'][0]

    trajectory = {
        'times': np.array(times),
        'poses': np.array(poses)
    }

    return trajectory
def read_data_generate_noise(filename, init_pose, noise_level=0.1):
    times = []
    poses = []

    with open(filename, 'r') as file:
        lines = file.readlines()

        for line in lines:
            parts = line.strip().split(', ')
            pose_timestamp = int(parts[0])
            pose = {
                'position': {
                    'x': float(parts[1]) - init_pose['position']['x'],
                    'y': float(parts[2]) - init_pose['position']['y'],
                    'z': float(parts[3]) - init_pose['position']['z'],
                },
                'orientation': {
                    'x': float(parts[4]),
                    'y': float(parts[5]),
                    'z': float(parts[6]),
                    'w': float(parts[7]),
                }
            }
            times.append(pose_timestamp)
            poses.append(pose)
    
    trajectory_gt = {
        'times': np.array(times),
        'poses': poses
    }

    trajectory_estimated = {'poses': []}
    for i, pose in enumerate(trajectory_gt['poses']):
        if i == 0:
            noisy_pose = {
                'position': {
                    'x': pose['position']['x'],
                    'y': pose['position']['y'],
                    'z': pose['position']['z'],
                },
                'orientation': pose['orientation']
            }
        else:
            prev_pose = trajectory_estimated['poses'][-1]
            distance = np.linalg.norm([
                pose['position']['x'] - trajectory_gt['poses'][i-1]['position']['x'],
                pose['position']['y'] - trajectory_gt['poses'][i-1]['position']['y'],
                pose['position']['z'] - trajectory_gt['poses'][i-1]['position']['z']
            ])
            
            if distance > 0.2:
                print("True")
                noise_level += 0.0015
                noisy_pose = {
                    'position': {
                        'x': prev_pose['position']['x'] + (pose['position']['x'] - trajectory_gt['poses'][i-1]['position']['x']) + np.random.normal(0, noise_level),
                        'y': prev_pose['position']['y'] + (pose['position']['y'] - trajectory_gt['poses'][i-1]['position']['y']) + np.random.normal(0, noise_level),
                        'z': prev_pose['position']['z'] + (pose['position']['z'] - trajectory_gt['poses'][i-1]['position']['z']) + np.random.normal(0, noise_level),
                    },
                    'orientation': pose['orientation']  # 保持 orientation 不变，或根据需要添加噪声
                }
            else:
                print("False")
                noisy_pose = {
                    'position': {
                        'x': prev_pose['position']['x'] + (pose['position']['x'] - trajectory_gt['poses'][i-1]['position']['x']),
                        'y': prev_pose['position']['y'] + (pose['position']['y'] - trajectory_gt['poses'][i-1]['position']['y']),
                        'z': prev_pose['position']['z'] + (pose['position']['z'] - trajectory_gt['poses'][i-1]['position']['z']),
                    },
                    'orientation': pose['orientation']  # 保持 orientation 不变，或根据需要添加噪声
                }
        trajectory_estimated['poses'].append(noisy_pose)

    return trajectory_estimated

class PoseEstimator:
    def __init__(self, image_dir, pose_filename, traj_gt_filename, start_index=0):
        self.image_dir = image_dir
        
        self.index = start_index
        self.image_N1 = None  
        self.image_N2 = None  
        self.pose_N1 = None
        self.T1 = None
        self.pose_N2 = None
        self.T2 = None
        # BFMatcher进行特征点匹配
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)

        self.estimated_pose = None
        
        # visualize
        self.img_match = None

        self.bridge = CvBridge()
        self.image_pub_1 = rospy.Publisher("/original_sonar_frame", Image, queue_size=10)
        self.image_pub_2 = rospy.Publisher("/selected_matches", Image, queue_size=10)
        self.pose_est_pub = rospy.Publisher("/pose_est", PoseArray, queue_size=10)
        self.traj_est_pub = rospy.Publisher("/trajectory_est", Path, queue_size=10)
        self.traj_gt_pub = rospy.Publisher("/trajectory_gt", Path, queue_size=10)
        
        self.init_pose = None
        with open(traj_gt_filename, 'r') as file:
            line = file.readlines()[0]
            parts = line.strip().split(', ')
            self.init_timestamp = int(parts[0])
            self.init_pose = {
                'position': {
                    'x': float(parts[1]),
                    'y': float(parts[2]),
                    'z': float(parts[3]),
                },
                'orientation': {
                    'x': float(parts[4]),
                    'y': float(parts[5]),
                    'z': float(parts[6]),
                    'w': float(parts[7]),
                }
            }
            
        self.poses = read_data_generate_noise(pose_filename, self.init_pose)
        self.trajectory = read_data(traj_gt_filename, self.init_pose)
        
            
        self.anp_solver = AnPAlgorithm()

    def publish_images(self):
        sonar_img_msg = self.bridge.cv2_to_imgmsg(cv2.cvtColor(self.image_N1, cv2.COLOR_BGR2RGB), "rgb8")
        matches_img_msg = self.bridge.cv2_to_imgmsg(cv2.cvtColor(self.img_match, cv2.COLOR_BGR2RGB), "rgb8")
        
        self.image_pub_1.publish(sonar_img_msg)
        self.image_pub_2.publish(matches_img_msg)

    def publish_pose_est(self):
        poses = self.poses['poses'][:self.index]
        
        pose_array_msg = PoseArray()
        pose_array_msg.header.stamp = rospy.Time.now()
        pose_array_msg.header.frame_id = "map"

        for pose in poses:
            ros_pose = Pose()
            ros_pose.position = Point(
                pose['position']['x'],
                pose['position']['y'],
                pose['position']['z']
            )
            ros_pose.orientation = Quaternion(
                pose['orientation']['x'],
                pose['orientation']['y'],
                pose['orientation']['z'],
                pose['orientation']['w']
            )
            pose_array_msg.poses.append(ros_pose)

        self.pose_est_pub.publish(pose_array_msg)
        
        ################
        
        path_msg = Path()
        path_msg.header.stamp = rospy.Time.now()
        path_msg.header.frame_id = "map"

        for pose in poses:
            pose_stamped = PoseStamped()
            pose_stamped.header.stamp = rospy.Time.now()
            pose_stamped.header.frame_id = "map"
            pose_stamped.pose.position.x = pose['position']['x']
            pose_stamped.pose.position.y = pose['position']['y']
            pose_stamped.pose.position.z = pose['position']['z']
            pose_stamped.pose.orientation.x = pose['orientation']['x']
            pose_stamped.pose.orientation.y = pose['orientation']['y']
            pose_stamped.pose.orientation.z = pose['orientation']['z']
            pose_stamped.pose.orientation.w = pose['orientation']['w']

            path_msg.poses.append(pose_stamped)

        self.traj_est_pub.publish(path_msg)
    
    def publish_traj_gt(self):
        path_msg = Path()
        path_msg.header.stamp = rospy.Time.now()
        path_msg.header.frame_id = "map"

        for pose in self.trajectory['poses']:
            pose_stamped = PoseStamped()
            pose_stamped.header.stamp = rospy.Time.now()
            pose_stamped.header.frame_id = "map"
            pose_stamped.pose.position.x = pose['position']['x']
            pose_stamped.pose.position.y = pose['position']['y']
            pose_stamped.pose.position.z = pose['position']['z']
            pose_stamped.pose.orientation.x = pose['orientation']['x']
            pose_stamped.pose.orientation.y = pose['orientation']['y']
            pose_stamped.pose.orientation.z = pose['orientation']['z']
            pose_stamped.pose.orientation.w = pose['orientation']['w']

            path_msg.poses.append(pose_stamped)

        self.traj_gt_pub.publish(path_msg)

    
    def visualize(self):
        self.publish_images()
        self.publish_traj_gt()
        self.publish_pose_est()
        return

    def akaze_feature_matching(self):
        # 初始化A-KAZE检测器
        akaze = cv2.AKAZE_create()

        # 检测特征点和描述子
        kp1, des1 = akaze.detectAndCompute(self.image_N1, None)
        kp2, des2 = akaze.detectAndCompute(self.image_N2, None) # kp2[0-334].pt 

        # 使用BFMatcher进行特征点匹配
        matches = self.bf.knnMatch(des1, des2, k=2)  # des1 (337, 61) des2 (334, 61)

        # 距离比测试
        good_matches = []
        for m, n in matches:
            if m.distance < 0.8 * n.distance:
                good_matches.append(m)

        # 提取匹配点
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        # 使用RANSAC估计单应性矩阵并过滤匹配点
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 10.0)
        matches_mask = mask.ravel().tolist()
        
        p_s_n1 = src_pts[mask.ravel() == 1]
        p_s_n2 = dst_pts[mask.ravel() == 1]

        # 可视化匹配结果
        # draw_params = dict(matchColor=(0, 255, 0), singlePointColor=None, matchesMask=matches_mask, flags=2)
        # img_match = cv2.drawMatches(self.image_N1, kp1, self.image_N2, kp2, good_matches, None, **draw_params)
        # cv2.imshow("Matches", img_match)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return p_s_n1, p_s_n2
    
    def main_process(self, step=1):
        rospy.init_node('pose_estimator')
        rate = rospy.Rate(1)
            
        image_files = sorted([f for f in os.listdir(self.image_dir) if f.endswith('.jpg')],
                             key=lambda x: int(x.split('_')[0]))
        
        while not rospy.is_shutdown():
            if self.index >= len(image_files) - 1:
                break
            ## STEP1 LOAD DATA 
            filename_N1 = image_files[self.index]
            filename_N2 = image_files[self.index + 1]

            image_path_N1 = os.path.join(self.image_dir, filename_N1)
            self.image_N1 = cv2.imread(image_path_N1)
            image_path_N2 = os.path.join(self.image_dir, filename_N2)
            self.image_N2 = cv2.imread(image_path_N2)

            # 打印第 N 帧的姿态数据
            # pose_timestamp_N1 = self.poses['times'][self.index] 
            self.pose_N1 = self.poses['poses'][self.index] 
            self.T1 = pose_to_transform_matrix(self.pose_N1)
            self.pose_N2 = self.poses['poses'][self.index+step] 
            self.T2 = pose_to_transform_matrix(self.pose_N2)
            
            
            ## STEP2 FEATURE MATCH  # (798, 512, 3)
            p_s_n1, p_s_n2 = self.akaze_feature_matching()
            # p_s_n1 = p_s_n1[0:1]
            # p_s_n2 = p_s_n2[0:1]
            # print(p_s_n1)
            # print(p_s_n2)
            # print(self.image_N1.shape)
            thetas1, distances1 = convert_points_to_polar(p_s_n1, self.image_N1.shape[1], self.image_N1.shape[0])
            thetas2, distances2 = convert_points_to_polar(p_s_n2, self.image_N1.shape[1], self.image_N1.shape[0])
            # print(thetas1)
            # print(distances1)
            # print(thetas2)
            # print(distances2)
            
            # IMG SHOW
            # 创建 DMatch 对象列表，只保留通过 RANSAC 筛选后的匹配
            selected_matches = [cv2.DMatch(_queryIdx=i, _trainIdx=i, _imgIdx=0, _distance=0) for i in range(len(p_s_n1))]
            # 创建 KeyPoint 对象列表
            kp1 = [cv2.KeyPoint(x=pt[0][0], y=pt[0][1], _size=1) for pt in p_s_n1]
            kp2 = [cv2.KeyPoint(x=pt[0][0], y=pt[0][1], _size=1) for pt in p_s_n2]
            # 可视化匹配结果
            self.img_match = cv2.drawMatches(self.image_N1, kp1, self.image_N2, kp2, selected_matches, None, matchColor=(0, 255, 0), singlePointColor=None, matchesMask=None, flags=2)
            # cv2.imshow("Selected Matches", img_match)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            
            
            print(f"Position N: {self.pose_N1}")
            print(f"Position N+1: {self.pose_N2}")
            print(self.index)
            
            # ## STEP3 TRI
            P_W = sonar_triangulation(self.T1, self.T2, distances1.reshape((-1, 1)), thetas1.reshape((-1, 1)), distances2.reshape((-1, 1)), thetas2.reshape((-1, 1)))
            
            self.visualize()
            # ## STEP4 ANP
            # # 计算 t_s 和 R_SW_Noise_my
            # P_SIs = np.vstack([
            #     distances1 * np.cos(thetas1),
            #     distances1 * np.sin(thetas1)
            # ])
            # t_s_cal, R_sw_cal = self.anp_solver.compute_t_R(P_SIs, P_W.T)

            # print("t_s_cal: \n", t_s_cal)
            # print("R_sw_cal: \n", R_sw_cal)  
            # print("#######################")  
            self.index += step
            rate.sleep()
            

if __name__ == '__main__':
    image_dir = './rec6/sonar_image'  # 请根据实际目录修改
    pose_filename = './rec6/pose_data.txt'  # 请根据实际文件路径修改
    # traj_gt = get_trajectory('./forward1/pose_data_continuous.txt' )
    traj_gt_filename = './rec6/pose_data_continuous.txt'
    estimator = PoseEstimator(image_dir, pose_filename, traj_gt_filename, start_index=0)
    estimator.main_process()

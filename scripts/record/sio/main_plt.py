import cv2
import os
import numpy as np
from tri import sonar_triangulation
from utils import pose_to_transform_matrix
from anp import AnPAlgorithm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.gridspec as gridspec

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

def get_trajectory(filename):
        x_data = []
        y_data = []
        z_data = []

        with open(filename, 'r') as file:
            lines = file.readlines()

            for line in lines:
                parts = line.strip().split(', ')
                x = float(parts[1])
                y = float(parts[2])
                z = float(parts[3])

                x_data.append(x)
                y_data.append(y)
                z_data.append(z)

        trajectory = {
            'x': np.array(x_data),
            'y': np.array(y_data),
            'z': np.array(z_data)
        }
        poses = np.vstack((x_data, y_data, z_data)).T  # 生成形状为 (n, 3) 的 numpy array

        return trajectory

class PoseEstimator:
    def __init__(self, image_dir, pose_filename, traj_gt,  start_index=18):
        self.image_dir = image_dir
        
        self.trajectory = traj_gt
        self.start_index = start_index
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
        
        with open(pose_filename, 'r') as file:
            self.pose_lines = file.readlines()
            
        self.anp_solver = AnPAlgorithm()

    def read_pose(self, index):
        if index < len(self.pose_lines):
            pose_line = self.pose_lines[index].strip()
            parts = pose_line.split(', ')
            pose_timestamp = int(parts[0])
            pose = {
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
            return pose_timestamp, pose
        else:
            return None, None
        
    def visualize(self):
        if self.trajectory is None:
            print("No trajectory data available.")
            return

        fig = plt.figure(figsize=(60, 15))
        gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1.5, 3])  # 设置比例为 1:2

        # 显示cv2图像
        ax1 = fig.add_subplot(gs[0])
        sonar_img = cv2.cvtColor(self.image_N1, cv2.COLOR_BGR2RGB)  # 转换为RGB格式以便matplotlib显示
        ax1.imshow(sonar_img)
        ax1.axis('off')
        ax1.set_title('Original Sonar Frame')
        
        # 显示cv2图像
        ax2 = fig.add_subplot(gs[1])
        img_match_rgb = cv2.cvtColor(self.img_match, cv2.COLOR_BGR2RGB)  # 转换为RGB格式以便matplotlib显示
        ax2.imshow(img_match_rgb)
        ax2.axis('off')
        ax2.set_title('Selected Matches')
        
        # 绘制3D轨迹
        ax3 = fig.add_subplot(gs[2], projection='3d')
        ax3.plot(self.trajectory['x'], self.trajectory['y'], self.trajectory['z'], label='3D Trajectory')
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')
        ax3.set_zlabel('Z')
        ax3.legend()

         # 定义相机视场参数
        cam_pos = [self.trajectory['x'][-1], self.trajectory['y'][-1], self.trajectory['z'][-1]]  # 相机位置为轨迹的最后一个点
        fov_height = 0.4  # 锥体高度

        fov_width = 1  # 棱锥底面宽度
        fov_length = 0.0005  # 棱锥底面长度
        # 计算棱锥的顶点和底面点
        apex = cam_pos
        direction = np.array([1, 0, 0])  # 假设相机朝向x轴正方向
        apex_to_base = direction * fov_height
        base_center = apex + apex_to_base

        # 生成棱锥底面矩形的四个角点
        base_rect_x = np.array([base_center[0], base_center[0], base_center[0], base_center[0]])
        base_rect_y = np.array([base_center[1] - fov_width / 2, base_center[1] + fov_width / 2, base_center[1] + fov_width / 2, base_center[1] - fov_width / 2])
        base_rect_z = np.array([base_center[2] - fov_length / 2, base_center[2] - fov_length / 2, base_center[2] + fov_length / 2, base_center[2] + fov_length / 2])

        # 创建棱锥的面
        vertices = [
            [apex, [base_rect_x[0], base_rect_y[0], base_rect_z[0]], [base_rect_x[1], base_rect_y[1], base_rect_z[1]]],
            [apex, [base_rect_x[1], base_rect_y[1], base_rect_z[1]], [base_rect_x[2], base_rect_y[2], base_rect_z[2]]],
            [apex, [base_rect_x[2], base_rect_y[2], base_rect_z[2]], [base_rect_x[3], base_rect_y[3], base_rect_z[3]]],
            [apex, [base_rect_x[3], base_rect_y[3], base_rect_z[3]], [base_rect_x[0], base_rect_y[0], base_rect_z[0]]],
            [[base_rect_x[0], base_rect_y[0], base_rect_z[0]], [base_rect_x[1], base_rect_y[1], base_rect_z[1]], [base_rect_x[2], base_rect_y[2], base_rect_z[2]], [base_rect_x[3], base_rect_y[3], base_rect_z[3]]]
        ]
        print(vertices)
        # 绘制棱锥
        ax3.add_collection3d(Poly3DCollection(vertices, facecolors='r', linewidths=1, edgecolors='r', alpha=0.25))

        plt.show()

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
        image_files = sorted([f for f in os.listdir(self.image_dir) if f.endswith('.jpg')],
                             key=lambda x: int(x.split('_')[0]))
        index = self.start_index

        while index < len(image_files) - 1:
            ## STEP1 LOAD DATA 
            
            filename_N1 = image_files[index]
            filename_N2 = image_files[index + 1]

            image_path_N1 = os.path.join(self.image_dir, filename_N1)
            self.image_N1 = cv2.imread(image_path_N1)
            image_path_N2 = os.path.join(self.image_dir, filename_N2)
            self.image_N2 = cv2.imread(image_path_N2)

            # 打印第 N 帧的姿态数据
            pose_timestamp_N1, self.pose_N1 = self.read_pose(index)
            self.T1 = pose_to_transform_matrix(self.pose_N1)
            pose_timestamp_N2, self.pose_N2 = self.read_pose(index+step)
            self.T2 = pose_to_transform_matrix(self.pose_N2)
            if self.pose_N1:
                # print(f"Image Index N: {index}")
                # print(f"Pose Timestamp N: {pose_timestamp_N}")
                print(f"Position N: {self.pose_N1}")
                # print(f"Orientation N: {self.pose_N1['orientation']}")
            else:
                print(f"No pose data found for image index N: {index}")
            index += step
            
            
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
            
            self.visualize()
            
            # ## STEP3 TRI
            # P_W = sonar_triangulation(self.T1, self.T2, distances1.reshape((-1, 1)), thetas1.reshape((-1, 1)), distances2.reshape((-1, 1)), thetas2.reshape((-1, 1)))
            
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
            cv2.destroyAllWindows()

if __name__ == '__main__':
    image_dir = './forward1/sonar_image'  # 请根据实际目录修改
    pose_filename = './forward1/pose_data.txt'  # 请根据实际文件路径修改
    traj_gt = get_trajectory('./forward1/pose_data_continuous.txt' )
    estimator = PoseEstimator(image_dir, pose_filename, traj_gt, start_index=0)
    estimator.main_process()

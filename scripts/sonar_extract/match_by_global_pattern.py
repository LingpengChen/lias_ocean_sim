import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

class PointMatcher:
    def __init__(self):
        self.image_size = (800, 800)  # 可视化图像大小
        
    def match_by_global_pattern(self, prev_candidates, curr_candidates):
        """
        使用整体模式匹配，考虑旋转和尺度变化
        Args:
            prev_candidates: 前一帧的点坐标列表 [(x1,y1), (x2,y2), ...]
            curr_candidates: 当前帧的点坐标列表 [(x1,y1), (x2,y2), ...]
        Returns:
            matches: 匹配对列表 [(prev_idx, curr_idx), ...]
        """
        if len(prev_candidates) != len(curr_candidates):
            return None
        
        prev_points = np.array(prev_candidates)
        curr_points = np.array(curr_candidates)
        
        # 1. 计算质心和归一化
        prev_center = np.mean(prev_points, axis=0)
        curr_center = np.mean(curr_points, axis=0)
        
        prev_normalized = prev_points - prev_center
        curr_normalized = curr_points - curr_center
        
        # 2. 计算尺度
        prev_scale = np.mean(np.linalg.norm(prev_normalized, axis=1))
        curr_scale = np.mean(np.linalg.norm(curr_normalized, axis=1))
        
        # 尺度归一化
        if prev_scale > 0 and curr_scale > 0:
            prev_normalized /= prev_scale
            curr_normalized /= curr_scale
        
        # 3. 考虑多个可能的旋转角度
        best_cost = float('inf')
        best_matches = None
        angles = np.linspace(0, 2*np.pi, 36)  # 每10度测试一次
        
        for angle in angles:
            rotation_matrix = np.array([
                [np.cos(angle), -np.sin(angle)],
                [np.sin(angle), np.cos(angle)]
            ])
            
            rotated_curr = np.dot(curr_normalized, rotation_matrix.T)
            
            # 构建成本矩阵
            cost_matrix = np.zeros((len(prev_normalized), len(rotated_curr)))
            for i, p1 in enumerate(prev_normalized):
                for j, p2 in enumerate(rotated_curr):
                    cost_matrix[i,j] = np.linalg.norm(p1 - p2)
            
            # 使用匈牙利算法找到最优匹配
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            total_cost = cost_matrix[row_ind, col_ind].sum()
            
            if total_cost < best_cost:
                best_cost = total_cost
                best_matches = list(zip(row_ind, col_ind))
        
        return best_matches

    def visualize_matches(self, prev_candidates, curr_candidates, matches):
        """
        可视化匹配结果
        """
        # 创建黑色背景图像
        vis_img = np.zeros((self.image_size[0], self.image_size[1] * 2, 3), dtype=np.uint8)
        
        # 在左半边绘制前一帧的点
        for x, y in prev_candidates:
            cv2.circle(vis_img, (int(x), int(y)), 5, (0, 255, 0), -1)  # 绿色
            
        # 在右半边绘制当前帧的点
        width = self.image_size[1]
        for x, y in curr_candidates:
            cv2.circle(vis_img, (int(x + width), int(y)), 5, (0, 0, 255), -1)  # 红色
            
        # 绘制匹配线
        if matches is not None:
            for prev_idx, curr_idx in matches:
                pt1 = (int(prev_candidates[prev_idx][0]), int(prev_candidates[prev_idx][1]))
                pt2 = (int(curr_candidates[curr_idx][0] + width), int(curr_candidates[curr_idx][1]))
                color = tuple(map(int, np.random.randint(64, 255, 3)))  # 随机颜色
                cv2.line(vis_img, pt1, pt2, color, 2)
        
        return vis_img

def test_point_matcher():
    def create_F_shape_points(base_x, base_y, scale=1.0, noise=1):
        """
        创建F形状的点分布（8个点）
        """
        # F形状的基本模式
        base_pattern = [
            (0, 0),    # 左上
            (0, 50),   # 左中
            (0, 100),  # 左下
            (30, 0),   # 上横线中点
            (60, 0),   # 上横线末端
            (30, 50),  # 中横线中点
            (45, 50),  # 中横线末端
            (15, 25),  # 中心点
        ]
        
        points = []
        for x, y in base_pattern:
            # 应用缩放
            scaled_x = x * scale
            scaled_y = y * scale
            
            # 添加噪声
            noisy_x = base_x + scaled_x + np.random.normal(0, noise)
            noisy_y = base_y + scaled_y + np.random.normal(0, noise)
            
            points.append((int(noisy_x), int(noisy_y)))
            
        return points

    def create_test_cases():
        """
        创建各种测试案例
        """
        test_cases = []
        
        # 案例1：轻微移动
        prev_points = create_F_shape_points(300, 300, scale=1.0)
        curr_points = create_F_shape_points(320, 310, scale=1.0, noise=3)
        test_cases.append((prev_points, curr_points, "Small Movement"))
        
        # 案例2：轻微旋转（通过矩阵变换）
        angle = np.pi/12  # 15度
        rotation_matrix = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)]
        ])
        prev_points = create_F_shape_points(300, 300, scale=1.0)
        base_points = create_F_shape_points(300, 300, scale=1.0, noise=0)
        curr_points = []
        for x, y in base_points:
            point = np.dot(rotation_matrix, np.array([x-300, y-300]))
            curr_points.append((int(point[0] + 300), int(point[1] + 300)))
        test_cases.append((prev_points, curr_points, "Rotation"))
        
        # 案例3：缩放
        prev_points = create_F_shape_points(300, 300, scale=1.0)
        curr_points = create_F_shape_points(300, 300, scale=1.2, noise=3)
        test_cases.append((prev_points, curr_points, "Scaling"))
        
        return test_cases

    # 运行测试
    matcher = PointMatcher()
    test_cases = create_test_cases()
    
    # 创建一个图像来显示所有测试案例
    plt.figure(figsize=(15, 15))
    
    for idx, (prev_points, curr_points, title) in enumerate(test_cases):
        # 执行匹配
        matches = matcher.match_by_global_pattern(prev_points, curr_points)
        
        # 可视化结果
        vis_img = matcher.visualize_matches(prev_points, curr_points, matches)
        
        # 显示结果
        plt.subplot(len(test_cases), 1, idx+1)
        plt.imshow(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
        plt.title(title)
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def visualize_single_frame(points, title="Points Distribution"):
    """
    可视化单帧点分布
    """
    img = np.zeros((800, 800, 3), dtype=np.uint8)
    
    # 绘制点
    for x, y in points:
        cv2.circle(img, (int(x), int(y)), 5, (0, 255, 0), -1)
        
    # 绘制连线以显示结构
    for i in range(len(points)-1):
        pt1 = (int(points[i][0]), int(points[i][1]))
        pt2 = (int(points[i+1][0]), int(points[i+1][1]))
        cv2.line(img, pt1, pt2, (0, 128, 255), 1)
    
    plt.figure(figsize=(8, 8))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    # 显示单个F形状的点分布

    test_point_matcher()
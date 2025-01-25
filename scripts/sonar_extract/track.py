import numpy as np
import matplotlib.pyplot as plt
import cv2
class Point:
    def __init__(self, x, y, id=None):
        self.x = x
        self.y = y
        self.id = id

from scipy.optimize import linear_sum_assignment

class Tracker:
    def __init__(self):
        self.previous_points = []
        
    def sort_by_previous_positions(self, candidates):
        if not self.previous_points:
            result = candidates[:8] if len(candidates) >= 8 else candidates
            for i, point in enumerate(result):
                point.id = i
            self.previous_points = result
            return result
        
        # 转换点为numpy数组格式
        prev_pts = np.float32([[p.x, p.y] for p in self.previous_points])
        curr_pts = np.float32([[p.x, p.y] for p in candidates])
        
        # 使用RANSAC找到单应性矩阵
        H, mask = cv2.findHomography(prev_pts, curr_pts, cv2.RANSAC, 5.0)
        print(H)
        # 如果找到有效的单应性矩阵
        if H is not None:
            # 预测前一帧点在当前帧的位置
            predicted_pts = cv2.perspectiveTransform(prev_pts.reshape(-1, 1, 2), H).reshape(-1, 2)
            
            # 构建cost matrix
            cost_matrix = np.zeros((len(self.previous_points), len(candidates)))
            for i, pred_pt in enumerate(predicted_pts):
                for j, candidate in enumerate(candidates):
                    dist = np.sqrt((candidate.x - pred_pt[0])**2 + (candidate.y - pred_pt[1])**2)
                    cost_matrix[i,j] = dist if dist < 50 else 1000000
                    
        # 使用匈牙利算法进行匹配
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        sorted_candidates = []
        for i, j in zip(row_ind, col_ind):
            if cost_matrix[i,j] < 50:
                candidates[j].id = self.previous_points[i].id  # 保持原有编号
                sorted_candidates.append(candidates[j])

        if len(sorted_candidates) < 8:
            used_candidates = set(sorted_candidates)
            remaining_candidates = [c for c in candidates if c not in used_candidates]
            
            if sorted_candidates:
                center_x = sum(c.x for c in sorted_candidates) / len(sorted_candidates)
                center_y = sum(c.y for c in sorted_candidates) / len(sorted_candidates)
                
                remaining_candidates.sort(key=lambda x: 
                    np.sqrt((x.x - center_x)**2 + (x.y - center_y)**2))
                
                # 为新添加的点分配新的编号
                used_ids = {p.id for p in sorted_candidates}
                next_id = 0
                for candidate in remaining_candidates[:8-len(sorted_candidates)]:
                    while next_id in used_ids:
                        next_id += 1
                    candidate.id = next_id
                    sorted_candidates.append(candidate)
        
        self.previous_points = sorted_candidates
        return sorted_candidates

def generate_test_points(num_points=8, noise_level=5):
    """生成正方形网格测试点"""
    points = []
    start_x, start_y = 50, 50  # 起始位置
    spacing = 20  # 点之间的间距
    
    # 计算行列数（取近似的平方根作为边长）
    side_length = int(np.sqrt(num_points))
    
    # 按行列生成点
    for i in range(2):
        for j in range(4):
            x = start_x + i * spacing
            y = start_y + j * spacing
            points.append(Point(x, y))
            
    return points

def add_noise_and_movement(points, dx=5, dy=3, noise_level=0.5):
    """添加噪声和移动"""
    noisy_points = []
    for p in points:
        new_x = p.x + dx + np.random.normal(0, noise_level)
        new_y = p.y + dy + np.random.normal(0, noise_level)
        noisy_points.append(Point(new_x, new_y))
    return noisy_points

def plot_points(points, color='b', marker='o', show_id=True):
    """绘制点和编号"""
    x = [p.x for p in points]
    y = [p.y for p in points]
    plt.scatter(x, y, c=color, marker=marker)
    if show_id:
        for p in points:
            if hasattr(p, 'id') and p.id is not None:
                plt.annotate(str(p.id), (p.x, p.y), 
                           xytext=(5, 5), textcoords='offset points')

import copy
# 运行测试
if __name__ == "__main__":
    plt.figure(figsize=(20, 5))
    tracker = Tracker()
    
    # 生成初始点
    initial_points = generate_test_points()
    candidates = initial_points
    tracked_points = tracker.sort_by_previous_positions(candidates)
    
    # 跟踪并绘制跟踪结果
    plot_points(candidates, color='r', marker='x', show_id=False)
    plot_points(tracked_points, color='b', marker='o', show_id=True)
    
    # extra_noise_points = [Point(np.random.uniform(0, 200), np.random.uniform(0, 200)) 
    #                     for _ in range(4)]  # 添加4个随机噪声点
    # candidates.extend(extra_noise_points)
    
    # 测试多个帧
    for i in range(3):
        plt.subplot(1, 3, i+1)
        
        # 生成当前帧的候选点（添加一些额外的噪声点）
        # candidates_copy = copy.deepcopy(candidates)
        candidates = add_noise_and_movement(candidates, dx=i*10, dy=i*15)
        np.random.shuffle(candidates)  # 添加这一行来打乱顺序

        # 绘制所有候选点
        plot_points(candidates, color='r', marker='x', show_id=False)
        
        # 跟踪并绘制跟踪结果
        tracked_points = tracker.sort_by_previous_positions(candidates)
        plot_points(tracked_points, color='b', marker='o', show_id=True)
        
        plt.title(f'Frame {i+1}')
        plt.xlim(0, 200)
        plt.ylim(0, 200)
        plt.grid(True)
    
    plt.tight_layout()
    plt.show()
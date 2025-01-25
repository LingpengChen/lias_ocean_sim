import cv2
import numpy as np
import matplotlib.pyplot as plt


def filter_nearest_points(candidates, k=8):
    if len(candidates) < k:
        return candidates
        
    from sklearn.neighbors import NearestNeighbors
    
    points = np.array(candidates)
    nbrs = NearestNeighbors(n_neighbors=min(k+1, len(points))).fit(points)
    distances, indices = nbrs.kneighbors(points)
    
    # 计算每个点到其他点的平均距离
    avg_distances = np.mean(distances[:, 1:], axis=1)  # 排除自身
    
    # 选择平均距离最小的点
    best_point_idx = np.argmin(avg_distances)
    
    # 获取距离最佳点最近的k个点的索引
    nearest_indices = indices[best_point_idx][1:]  # 排除自身
    filtered_candidates = [candidates[i] for i in nearest_indices]
    
    # 添加最佳点本身
    filtered_candidates.insert(0, candidates[best_point_idx])
    
    return filtered_candidates

class BallDetector:
    def __init__(self):
        self.prev_positions = None
        
    def detect_and_track_balls(self, image):
        # 确保图像是8位格式
        if image.dtype != np.uint8:
            image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        cv2.imshow('Ori', image)
        cv2.waitKey(0)
        # 1. 预处理
        blurred = cv2.GaussianBlur(image, (0,0), sigmaX=3, sigmaY=1)
        
        
        alpha = 5  # 对比度因子
        beta = 0    # 亮度调整
        blurred = cv2.convertScaleAbs(blurred, alpha=alpha, beta=beta)

        # clahe = cv2.createCLAHE(clipLimit=20.0, tileGridSize=(8,8))
        # blurred = clahe.apply(blurred)
        cv2.imshow('Gaussian', blurred)
        cv2.waitKey(0)
        
        # 2. 二值化 - 使用Otsu方法
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cv2.imshow('Binary View', binary)
        cv2.waitKey(0)
        
        # 3. 形态学操作
        kernel = np.ones((3,3), np.uint8)
        binary = cv2.dilate(binary, kernel)  # 直接使用膨胀操作
        cv2.imshow('dilate', binary)
        cv2.waitKey(0)
        
        # 4. 连通区域分析
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)
        
        # 5. 提取候选球
        candidates = []
        print(stats[:, cv2.CC_STAT_AREA])
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if 10 < area < 400:  # 面积阈值，需要根据实际情况调整
                x = int(centroids[i][0])
                y = int(centroids[i][1])
                w = stats[i, cv2.CC_STAT_WIDTH]
                h = stats[i, cv2.CC_STAT_HEIGHT]
                
                # 计算圆形度
                circularity = 4 * np.pi * area / (w * h)
                if circularity > 0.7:  # 圆形度阈值
                    candidates.append((x, y))
                    
        # clustered 8 points
        candidates = filter_nearest_points(candidates, k=8)
        # 6. 使用前一帧信息进行跟踪
        if self.prev_positions and len(self.prev_positions) == 8:
            candidates = self.sort_by_previous_positions(candidates, self.prev_positions)
        
        # 确保只返回8个点
        candidates = candidates[:8] if len(candidates) >= 8 else candidates
        
        # 更新前一帧位置
        self.prev_positions = candidates
        
        return candidates, binary
    
    def sort_by_previous_positions(self, candidates, prev_positions):
        sorted_candidates = []
        used_candidates = set()
        
        for prev_pos in prev_positions:
            min_dist = float('inf')
            best_candidate = None
            
            for candidate in candidates:
                if candidate not in used_candidates:
                    dist = np.sqrt((candidate[0] - prev_pos[0])**2 + 
                                 (candidate[1] - prev_pos[1])**2)
                    if dist < min_dist:
                        min_dist = dist
                        best_candidate = candidate
            
            if best_candidate:
                sorted_candidates.append(best_candidate)
                used_candidates.add(best_candidate)
        
        # 添加剩余的候选点
        for candidate in candidates:
            if candidate not in used_candidates:
                sorted_candidates.append(candidate)
        
        return sorted_candidates

def visualize_results(image, detected_points, binary):
    # 创建可视化图像
    vis_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    # 在原图上标记检测到的点
    for i, (x, y) in enumerate(detected_points):
        cv2.circle(vis_image, (x, y), 5, (0, 255, 0), -1)  # 绿色圆点
        cv2.putText(vis_image, str(i+1), (x+10, y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    # 确保所有图像都是3通道
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if len(binary.shape) == 2:    
        binary = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    # 水平拼接三张图像
    result = cv2.hconcat([image, binary, vis_image])

    # 创建窗口显示拼接结果
    cv2.namedWindow('Results', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Results', 4500, 1200)
    cv2.imshow('Results', result)

    # 等待按键
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    # 初始化检测器
    detector = BallDetector()
    
    # 读取图像
    for i in range(6):
        img = cv2.imread(f'output_{i}.png', cv2.IMREAD_GRAYSCALE)
        
        # 处理第一张图片
        points, binary = detector.detect_and_track_balls(img)
        visualize_results(img, points, binary)
        
        print("Image 2 points:", points)

if __name__ == "__main__":
    main()
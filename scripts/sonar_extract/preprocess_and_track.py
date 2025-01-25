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
        self.balls_num = 8
        
        self.prev_image = None
        self.image = None
        
        self.prev_descriptors = None
        self.prev_candidates = None
        
        self.akaze = cv2.AKAZE_create(
            threshold=0.0008,
            nOctaves=4,
            nOctaveLayers=4,
            diffusivity=cv2.KAZE_DIFF_PM_G2
        )
        

    
    def __extract_ball_candidates(self):    
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
  
        # 确保图像是8位格式
        if gray.dtype != np.uint8:
            gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        # Gaussian blurred and increase intensity
        blurred = cv2.GaussianBlur(gray, (0,0), sigmaX=3, sigmaY=1)
        blurred = cv2.convertScaleAbs(blurred, alpha=5, beta=0) # alpha对比度因子  # beta亮度调整
        
        # 2.Change into binary value 
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 3. Dilate
        kernel = np.ones((3,3), np.uint8)
        binary = cv2.dilate(binary, kernel)  # 直接使用膨胀操作
        
        # 4. 连通区域分析 get area
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
        
        return candidates
    
    def __get_descriptors(self, candidates):
                 # 为每个候选点提取AKAZE特征
        
        # 为每个候选点创建小的ROI来提取特征
        current_descriptors = []
        roi_size = 40  # ROI窗口大小
        
        for x, y in candidates:
            # 确保ROI在图像范围内
            x1 = max(0, x - roi_size//2)
            y1 = max(0, y - roi_size//2)
            x2 = min(self.image.shape[1], x + roi_size//2)
            y2 = min(self.image.shape[0], y + roi_size//2)
            
            roi = self.image[y1:y2, x1:x2]
            if roi.size == 0:  # 检查ROI是否为空
                continue
                
            # 检测关键点和描述符
            keypoints, descriptors = self.akaze.detectAndCompute(roi, None)
            if descriptors is not None and len(keypoints) > 0:
                # 使用最强的特征描述符
                current_descriptors.append(descriptors[0])
        return current_descriptors
    
    def visualize_matches(self):
        """
        Visualize the matching between previous frame and current frame
        Returns:
            visualization image with matching lines
        """
        if not hasattr(self, 'prev_image') or self.prev_image is None or \
        not hasattr(self, 'prev_candidates') or not self.prev_candidates or \
        not hasattr(self, 'candidates') or not self.candidates:
            return None
            
        # 创建拼接图像
        h, w, c= self.image.shape
        vis_img = np.zeros((h, w*2, c), dtype=np.uint8)
        vis_img[:, :w] = self.prev_image
        vis_img[:, w:] = self.image
        
        # 转换为彩色图像以绘制彩色线条
        # vis_img = cv2.cvtColor(vis_img, cv2.COLOR_GRAY2BGR)
        
        # 在两帧图像上画出特征点
        for x, y in self.prev_candidates:
            cv2.circle(vis_img, (x, y), 3, (0, 255, 0), -1)  # 绿色圆点表示前一帧的特征点
            
        for x, y in self.candidates:
            cv2.circle(vis_img, (x + w, y), 3, (0, 0, 255), -1)  # 红色圆点表示当前帧的特征点
        
        # 如果特征点数量相同，画出匹配线
        if len(self.prev_candidates) == len(self.candidates):
            for (x1, y1), (x2, y2) in zip(self.prev_candidates, self.candidates):
                # 绘制匹配线，使用随机颜色
                color = tuple(map(int, np.random.randint(0, 255, 3)))
                cv2.line(vis_img, (x1, y1), (x2 + w, y2), color, 1)
                
        # 添加帧号或其他信息
        cv2.putText(vis_img, "Previous Frame", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(vis_img, "Current Frame", (w + 10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        cv2.imshow("Matching Visualization", vis_img)
        cv2.waitKey(0)
        # return vis_img
    
    def detect_and_track_balls(self, image):
        # find clustered 8 points
        self.image = image
      
        candidates = self.__extract_ball_candidates()
        candidates = filter_nearest_points(candidates, k=self.balls_num)
        
        current_descriptors = self.__get_descriptors(candidates)


        # 如果是第一帧或没有特征点，保存当前特征并返回
        if self.prev_descriptors is None:
            self.prev_descriptors = current_descriptors
            self.prev_candidates = candidates
            return candidates

        else:
            # 将描述符列表转换为numpy数组
            current_desc_array = np.array(current_descriptors)
            prev_desc_array = np.array(self.prev_descriptors)
            
            matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_BRUTEFORCE_HAMMING)
            matches = matcher.knnMatch(prev_desc_array, current_desc_array, k=2 if len(current_descriptors) > 1 else 1)
            
            # 应用比率测试来筛选好的匹配
            good_matches = []
            for match in matches:
                # if len(match) == 2:
                #     m, n = match
                #     if m.distance < 0.7 * n.distance:  # Lowe's ratio test
                #         good_matches.append((m.queryIdx, m.trainIdx))
                # else:
                good_matches.append((match[0].queryIdx, match[0].trainIdx))
            
            # 根据匹配结果重新排序当前候选点
            if good_matches:
                sorted_candidates = []
                used_indices = set()
                
                for prev_idx, curr_idx in good_matches:
                    if curr_idx not in used_indices:
                        sorted_candidates.append(candidates[curr_idx])
                        used_indices.add(curr_idx)
                
                # 添加未匹配的候选点
                for i in range(len(candidates)):
                    if i not in used_indices:
                        sorted_candidates.append(candidates[i])
                
                candidates = sorted_candidates[:self.balls_num]
        
        # 更新上一帧的特征和候选点
        self.candidates = candidates
        if self.prev_image is not None:
            self.visualize_matches()
        
        self.prev_descriptors = current_descriptors
        self.prev_candidates = candidates
        self.prev_image = self.image
        
        return candidates
    
    def get_image(self):
        return self.image


import copy

def visualize_results(image, detected_points):
    # 创建可视化图像
    vis_image = copy.deepcopy(image)
    
    # 在原图上标记检测到的点
    for i, (x, y) in enumerate(detected_points):
        cv2.circle(vis_image, (x, y), 5, (0, 255, 0), -1)  # 绿色圆点
        cv2.putText(vis_image, str(i+1), (x+10, y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    # # 确保所有图像都是3通道
    # if len(image.shape) == 2:
    #     image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    # if len(binary.shape) == 2:    
    #     binary = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    # # 水平拼接三张图像
    result = cv2.hconcat([image, vis_image])

    # 创建窗口显示拼接结果
    cv2.namedWindow('Results', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Results', 2400, 1200)
    cv2.imshow('Results', result)

    # 等待按键
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    # 初始化检测器
    detector = BallDetector()
    
    # 读取图像
    for i in range(6):
        img = cv2.imread(f'output_{i}.png')

        points = detector.detect_and_track_balls(img)
        visualize_results(img, points)
        
        print("Image 2 points:", points)

if __name__ == "__main__":
    main()
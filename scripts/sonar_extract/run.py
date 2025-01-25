import cv2
import numpy as np
import glob
import pickle
from scipy.spatial.distance import cdist

class TemplatePointMatcher:
    def __init__(self):
        self.template_points = []
        self.selected_points = []
        self.current_image = None

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.selected_points) < 8:
                self.selected_points.append([x, y])
                # 在图像上画点
                cv2.circle(self.current_image, (x, y), 3, (0, 255, 0), -1)
                # 显示点的序号
                cv2.putText(self.current_image, str(len(self.selected_points)-1), 
                          (x+5, y+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.imshow('Image', self.current_image)

    def create_template(self, first_image_path):
        # 读取第一张图片
        self.current_image = cv2.imread(first_image_path)
        clone = self.current_image.copy()
        cv2.namedWindow('Image')
        cv2.setMouseCallback('Image', self.mouse_callback)

        while True:
            cv2.imshow('Image', self.current_image)
            key = cv2.waitKey(1) & 0xFF
            
            # 如果按 'r'，重置选择
            if key == ord('r'):
                self.current_image = clone.copy()
                self.selected_points = []
            # 如果按 'c'，确认选择
            elif key == ord('c') and len(self.selected_points) == 8:
                break

        cv2.destroyAllWindows()
        self.template_points = np.array(self.selected_points)
        return self.template_points

    def match_points(self, image_path):
        # 读取图片
        image = cv2.imread(image_path)
        # 转换为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 使用Harris角点检测
        corners = cv2.goodFeaturesToTrack(gray, 25, 0.01, 10)
        corners = np.int0(corners)
        detected_points = corners.reshape(-1, 2)

        # 如果检测到的点太少，返回全0数组
        if len(detected_points) < 8:
            return np.zeros((8, 2))

        # 计算检测点之间的距离矩阵
        template_distances = cdist(self.template_points, self.template_points)
        
        best_matches = []
        used_points = set()
        
        # 对每个模板点找最佳匹配
        for i, template_point in enumerate(self.template_points):
            min_diff = float('inf')
            best_match = None
            
            for detected_point in detected_points:
                if tuple(detected_point) in used_points:
                    continue
                    
                # 计算当前检测点到其他已匹配点的距离
                current_distances = []
                template_distances_i = []
                
                for j, matched in enumerate(best_matches):
                    if matched is not None:
                        current_distances.append(np.linalg.norm(detected_point - matched))
                        template_distances_i.append(template_distances[i][j])
                
                # 比较距离模式
                if len(current_distances) > 0:
                    diff = np.sum(np.abs(np.array(current_distances) - np.array(template_distances_i)))
                else:
                    diff = np.linalg.norm(detected_point - template_point)
                
                if diff < min_diff:
                    min_diff = diff
                    best_match = detected_point

            if best_match is not None and min_diff < 100:  # 阈值可调整
                best_matches.append(best_match)
                used_points.add(tuple(best_match))
            else:
                best_matches.append(np.array([0, 0]))

        return np.array(best_matches)

def main():
    # 获取所有图片路径
    image_paths = sorted(glob.glob('output_*.png'))
    
    if not image_paths:
        print("No images found!")
        return

    matcher = TemplatePointMatcher()
    
    # 使用第一张图片创建模板
    print("请在第一张图片上标记8个点，按'r'重置，按'c'确认")
    template = matcher.create_template(image_paths[0])
    
    # 保存模板点
    with open('template_points.pkl', 'wb') as f:
        pickle.dump(template, f)
    
    # 处理所有图片
    all_coordinates = []
    for image_path in image_paths:
        print(f"Processing {image_path}")
        coordinates = matcher.match_points(image_path)
        all_coordinates.append(coordinates)
        
        # 在图片上显示匹配结果
        img = cv2.imread(image_path)
        for i, point in enumerate(coordinates):
            if not (point == 0).all():
                cv2.circle(img, tuple(point), 3, (0, 255, 0), -1)
                cv2.putText(img, str(i), (point[0]+5, point[1]+5), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        cv2.imshow('Matches', img)
        cv2.waitKey(1000)  # 显示1秒
    
    cv2.destroyAllWindows()
    
    # 保存所有坐标
    with open('point_coordinates.pkl', 'wb') as f:
        pickle.dump(all_coordinates, f)
    
    # 打印结果
    for i, coords in enumerate(all_coordinates):
        print(f"\nImage {i} coordinates:")
        for j, point in enumerate(coords):
            print(f"Point {j}: {point}")

if __name__ == "__main__":
    main()
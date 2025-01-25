import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

def mutual_nearest_matches(pts1, pts2):
    # 计算距离矩阵
    distances = cdist(pts1, pts2)
    
    # 找到互为最近邻的点对
    matches = []
    for i in range(len(pts1)):
        nearest_j = np.argmin(distances[i])
        # 检查是否互为最近邻
        if i == np.argmin(distances[:, nearest_j]):
            matches.append((i, nearest_j))
    
    src_pts = np.float32([pts1[i] for i, j in matches])
    dst_pts = np.float32([pts2[j] for i, j in matches])
    
    return src_pts, dst_pts

# 创建两个示例图像框架
frame1 = np.zeros((400, 400), dtype=np.uint8)
frame2 = np.zeros((400, 400), dtype=np.uint8)

# 在frame1中创建10个源点
src_pts = np.array([
    [100, 100],
    [150, 100],
    [200, 100],
    [100, 150],
    [150, 150],
    [200, 150],
    [100, 200],
    [150, 200],
    [200, 200],
    [175, 175]
], dtype=np.float32)

# 定义一个变换矩阵（例如：平移+缩放+旋转）
true_H = np.array([
    [1, 0, 10],    # 缩放1.1倍，略微旋转，x方向平移10
    [-0, 1, 15],   # y方向平移15
    [0, 0, 1]
])

# 使用真实的变换矩阵生成目标点
src_pts_homog = np.hstack((src_pts, np.ones((src_pts.shape[0], 1))))  # 转换为齐次坐标
dst_pts_homog = np.dot(src_pts_homog, true_H.T)
dst_pts = (dst_pts_homog[:, :2] / dst_pts_homog[:, 2:3]).astype(np.float32)
# dst_pts = dst_pts_homog
np.random.shuffle(dst_pts)  # 添加这一行来打乱顺序


# 计算单应性矩阵
src_pts, dst_pts = mutual_nearest_matches(src_pts, dst_pts)
M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

# 可视化
plt.figure(figsize=(12, 5))

# 绘制frame1中的点
plt.subplot(121)
plt.title('Frame 1 - Source Points')
plt.scatter(src_pts[:, 0], src_pts[:, 1], c='blue', marker='o')
for i, point in enumerate(src_pts):
    plt.annotate(f'P{i}', (point[0], point[1]))
plt.xlim(0, 400)
plt.ylim(0, 400)
plt.gca().invert_yaxis()  # 反转y轴使其与图像坐标系一致

# 绘制frame2中的点
plt.subplot(122)
plt.title('Frame 2 - Destination Points')
plt.scatter(dst_pts[:, 0], dst_pts[:, 1], c='red', marker='o')
for i, point in enumerate(dst_pts):
    plt.annotate(f'P{i}', (point[0], point[1]))
    
# 使用找到的单应性矩阵变换源点
transformed_pts = cv2.perspectiveTransform(src_pts.reshape(-1, 1, 2), M)
transformed_pts = transformed_pts.reshape(-1, 2)
plt.scatter(transformed_pts[:, 0], transformed_pts[:, 1], c='green', marker='+', 
           label='Transformed Points')

plt.xlim(0, 400)
plt.ylim(0, 400)
plt.gca().invert_yaxis()
plt.legend()

plt.tight_layout()
plt.show()

# 打印单应性矩阵
print("Homography Matrix:")
print(M)

# 打印mask（表示哪些点是内点）
print("\nMask (inliers=1, outliers=0):")
print(mask.ravel())
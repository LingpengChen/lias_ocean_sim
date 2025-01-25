import cv2
import numpy as np

def extract_sonar_features(image):
    # 1. 基础预处理
    if image.dtype != np.uint8:
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    enhanced = cv2.convertScaleAbs(image, alpha=5, beta=0) # alpha对比度因子  # beta亮度调整
    

    # 4. A-KAZE特征提取
    akaze = cv2.AKAZE_create(
        threshold=0.0008,
        nOctaves=4,
        nOctaveLayers=4,
        diffusivity=cv2.KAZE_DIFF_PM_G2
    )
    keypoints, descriptors = akaze.detectAndCompute(image, None)
    
    # 5. 特征点过滤
    filtered_keypoints = []
    filtered_descriptors = []
    
    if len(keypoints) > 0:  # 确保有检测到特征点
        for idx, kp in enumerate(keypoints):
            # 响应强度和尺度的过滤
            if 8 < kp.size < 10:
                # 位置过滤（可选，根据声纳图像特性调整）
                x, y = kp.pt
                if y > image.shape[0] * 0.1:  # 过滤掉顶部10%区域的点
                    filtered_keypoints.append(kp)
                    if descriptors is not None:
                        filtered_descriptors.append(descriptors[idx])
    
    if len(filtered_descriptors) > 0:
        filtered_descriptors = np.array(filtered_descriptors)
    
    # 6. 可视化处理结果（调试用）
    debug_image = cv2.drawKeypoints(enhanced, filtered_keypoints, None,
                                  color=(0,255,0), 
                             flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    return enhanced, filtered_keypoints, filtered_descriptors, debug_image

# 使用示例
    
if __name__ == "__main__":
    img1 = cv2.imread('output_1.png', cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread('output_5.png', cv2.IMREAD_GRAYSCALE)

    # 处理第一张图片
    enhanced1, kpts1, desc1, debug1 = extract_sonar_features(img1)
    # 处理第二张图片
    enhanced2, kpts2, desc2, debug2 = extract_sonar_features(img2)



    combined_img = cv2.hconcat([debug1, debug2])
    cv2.imshow('Image View', combined_img)
    cv2.waitKey(0)
    
    
    
    # 特征匹配
    matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_BRUTEFORCE_HAMMING)
    matches = matcher.knnMatch(desc1, desc2, k=2)
    
    # 应用比率测试
    good_matches = []
    pts1 = []
    pts2 = []
    for m, n in matches:
        if m.distance < 0.95 * n.distance:
            good_matches.append(m)
            pts1.append(kpts1[m.queryIdx].pt)
            pts2.append(kpts2[m.trainIdx].pt)
    
    # # 转换点格式并RANSAC
    # pts1 = np.float32(pts1)
    # pts2 = np.float32(pts2)
    # H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
    
    # 可视化匹配结果
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    height = max(h1, h2)
    img1_resized = cv2.resize(img1, (int(w1 * height / h1), height))
    img2_resized = cv2.resize(img2, (int(w2 * height / h2), height))
    
    vis = np.zeros((height, img1_resized.shape[1] + img2_resized.shape[1]), np.uint8)
    vis[:, :img1_resized.shape[1]] = img1_resized
    vis[:, img1_resized.shape[1]:] = img2_resized
    vis_color = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
    
    # 绘制所有good_matches的匹配线
    for m in good_matches:
        p1 = tuple(map(int, kpts1[m.queryIdx].pt))
        p2 = tuple(map(int, (kpts2[m.trainIdx].pt[0] + img1_resized.shape[1], 
                           kpts2[m.trainIdx].pt[1])))
        cv2.line(vis_color, p1, p2, (0, 255, 0), 1)
        cv2.circle(vis_color, p1, 5, (0, 255, 0), -1)
        cv2.circle(vis_color, p2, 5, (0, 255, 0), -1)
    
    # 显示结果
    print(f'特征点数量: img1={len(kpts1)}, img2={len(kpts2)}')
    print(f'KNN匹配数量: {len(matches)}')
    print(f'good matches数量: {len(good_matches)}')
    cv2.imwrite('matching_result.jpg', vis_color)
    cv2.imshow('Matches', vis_color)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


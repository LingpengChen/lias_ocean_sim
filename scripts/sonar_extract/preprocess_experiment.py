#!/usr/bin/env python3
import cv2
import numpy as np

def extract_sonar_features(image):
    # 1. 预处理
    # 对比度增强
    # 确保图像是单通道的
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 确保图像是8位格式
    if image.dtype != np.uint8:
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    

    # 降噪
    # denoised = cv2.fastNlMeansDenoising(enhanced)
    # 1. 简单预处理
    # 高斯模糊去噪
    # denoised = cv2.GaussianBlur(image, (3, 13), 0)
    blurred = cv2.GaussianBlur(image, (0,0), sigmaX=5, sigmaY=2)

    # # 对比度增强
    # alpha = 3  # 对比度增强因子
    # beta = 0    # 亮度增加值
    # enhanced = cv2.convertScaleAbs(blurred, alpha=alpha, beta=beta)
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(10,10))
    enhanced = clahe.apply(blurred)
    
    # result = cv2.addWeighted(image, 1.0, blurred, -1.0, 0)
    combined_img = cv2.hconcat([image, blurred, enhanced])
    cv2.imshow('Image View', combined_img)
    cv2.waitKey(0)
    
    # 2. A-KAZE特征点提取
    akaze = cv2.AKAZE_create(
        threshold=0.001,  # 降低阈值以检测更多特征点
        nOctaves=4,
        nOctaveLayers=4,
    )
    keypoints = akaze.detect(enhanced, None)
    result = cv2.drawKeypoints(image, keypoints, None, 
                             color=(0,255,0), 
                             flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow('Image temp', result)
    cv2.waitKey(0)
    
    
    # 3. 基于强度和大小的过滤
    filtered_keypoints = []
    for kp in keypoints:
        # 可以根据实际情况调整这些阈值
        # if kp.response > 0.01:
        if kp.response > 0.001 and 10 < kp.size < 12:
            filtered_keypoints.append(kp)
    
    # 4. 绘制结果
    result = cv2.drawKeypoints(image, filtered_keypoints, None, 
                             color=(0,255,0), 
                             flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow('Image temp', result)
    cv2.waitKey(0)
    
    return filtered_keypoints, result


# 使用示例
# 读取图像时直接转换为灰度图
sonar_image = cv2.imread('output_5.png')

keypoints, result_image = extract_sonar_features(sonar_image)
combined_img = cv2.hconcat([sonar_image, result_image])
cv2.imshow('Image View', combined_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
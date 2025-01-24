#!/usr/bin/env python3
import cv2
import cv2.aruco as aruco
import numpy as np

# 创建 ChArUco board
dictionary = aruco.Dictionary_get(aruco.DICT_6X6_250)
board = aruco.CharucoBoard_create(
    squaresX=2,
    squaresY=2,
    squareLength=0.1,
    markerLength=0.05,
    dictionary=dictionary
)

# 生成板的图像
img = board.draw((800, 800))  # 生成800x800像素的图像

# 保存图像
cv2.imwrite('charuco_board_2_2.png', img)

# 显示图像
cv2.imshow('Charuco board', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("Charuco board has been saved as 'charuco_board.png'")
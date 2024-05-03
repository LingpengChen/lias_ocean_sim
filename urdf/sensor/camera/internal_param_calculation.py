import math

# 图像尺寸和视场角
width = 640
hfov_degrees = 90
hfov_radians = math.radians(hfov_degrees) # 1.5707
# focal_length = 320
# 计算焦距
focal_length = width / (2 * math.tan(hfov_radians / 2))
print(focal_length)

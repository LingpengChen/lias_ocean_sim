import math

def rad2deg(rad):
    return rad * 180 / math.pi

def deg2rad(deg):
    return deg * math.pi / 180

# 测试
rad = 0.174533
deg = rad2deg(rad)
print(f"{rad} 弧度 = {deg} 度")

test_deg = 30
test_rad = deg2rad(test_deg)
print(f"{test_deg} 度 = {test_rad} 弧度")
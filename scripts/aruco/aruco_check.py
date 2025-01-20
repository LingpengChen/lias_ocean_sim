import cv2.aruco as aruco
import cv2
dictionary = aruco.Dictionary_get(aruco.DICT_4X4_250)
board = aruco.CharucoBoard_create(
            squaresX=5, 
            squaresY=7,
            squareLength=0.04,
            markerLength=0.02,
            dictionary=dictionary
        )
# 检查所有需要的函数
functions_to_check = [
    'detectMarkers',
    'interpolateCornersCharuco',
    'estimatePoseCharucoBoard',
    'Dictionary_get',
    'CharucoBoard_create'
]

print("OpenCV version:", cv2.__version__)
print("\nChecking aruco functions:")
for func in functions_to_check:
    has_func = hasattr(aruco, func)
    print(f"aruco.{func}: {'✓' if has_func else '✗'}")

# 可以也检查一下你已创建的对象
print("\nChecking created objects:")
print(f"dictionary type: {type(dictionary)}")
print(f"board type: {type(board)}")

# 列出aruco模块的所有可用函数
print("\nAll available functions in aruco module:")
print([x for x in dir(aruco) if not x.startswith('_')])
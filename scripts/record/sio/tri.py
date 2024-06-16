import numpy as np

def sonar_triangulation(T1, T2, d1, theta1, d2, theta2):
    """
    T1: The first sonar pose w.r.t. the world frame, represented as a 4x4 matrix [R1, t1; 0, 1].
    T2: The second sonar pose w.r.t. the world frame, represented as a 4x4 matrix [R2, t2; 0, 1].
    d1: The distance measurement of the first sonar in meters.
    theta1: The angle measurement of the first sonar in radians.
    d2: The distance measurement of the second sonar in meters.
    theta2: The angle measurement of the second sonar in radians.
    Returns:
    P_W: The estimated coordinates of the 3D points in the world frame.
    """
    # 确保输入是numpy数组
    d1 = np.asarray(d1).reshape(-1)
    theta1 = np.asarray(theta1).reshape(-1)
    d2 = np.asarray(d2).reshape(-1)
    theta2 = np.asarray(theta2).reshape(-1)
    
    
    # 确保所有输入数组的长度相同
    if not (len(d1) == len(theta1) == len(d2) == len(theta2)):
        raise ValueError("All input arrays must have the same length")

    # Initialize a list to hold the results
    points_3d = []

    # Calculate the relative transformation
    T = np.linalg.inv(T2) @ T1
    R = T[:3, :3]
    t = T[:3, 3]

    r1 = R[0, :]
    r2 = R[1, :]

    err = False
    
    for i in range(len(d1)):
        # Solve the first set of linear equations for each point
        A1 = np.vstack([
            [np.tan(theta1[i]), -1, 0],
            [np.tan(theta2[i]) * r1 - r2],
            [t @ R]
        ])
        b1 = np.array([0, t[1] - np.tan(theta2[i]) * t[0], (d2[i]**2 - d1[i]**2 - np.linalg.norm(t)**2) / 2])

        x = np.linalg.inv(A1) @ b1

        # Solve the second set of linear equations for each point
        A2 = np.vstack([A1, x])
        b2 = np.append(b1, d1[i]**2)
        try:
            point_3d = np.linalg.inv(A2.T @ A2) @ A2.T @ b2
        except Exception as e:
            if not err:
                with open('error_file.txt', 'w') as file:
                    file.write(f"An error occurred: {str(e)}\n")
                    file.write(f"T1: {T1}\n")
                    file.write(f"T2: {T2}\n")
                    file.write(f"d1: {d1}\n")
                    file.write(f"theta1: {theta1}\n")
                    file.write(f"d2: {d2}\n")
                    file.write(f"theta2: {theta2}\n\n")
                    file.write(f"d1[i]: {d1[i]}\n")
                    file.write(f"theta1[i]: {theta1[i]}\n")
                    file.write(f"d2[i]: {d2[i]}\n")
                    file.write(f"theta2[i]: {theta2[i]}\n\n")

                    print("Error occurred and details are saved to error_file.txt")
                err = True
            else:
                with open('error_file.txt', 'a') as file:
                    file.write(f"d1[i]: {d1[i]}\n")
                    file.write(f"theta1[i]: {theta1[i]}\n")
                    file.write(f"d2[i]: {d2[i]}\n")
                    file.write(f"theta2[i]: {theta2[i]}\n\n")
        # Append the result to the list
        points_3d.append(point_3d)

    return np.array(points_3d)

# # 示例调用
# T1 = np.eye(4)  # 示例4x4变换矩阵
# T2 = np.eye(4)

# distances1 = np.array([1, 2, 3])
# thetas1 = np.array([0.1, 0.2, 0.3])
# distances2 = np.array([4, 5, 6])
# thetas2 = np.array([0.4, 0.5, 0.6])

# P_W = sonar_triangulation(T1, T2, distances1, thetas1, distances2, thetas2)
# print("3D Points in World Frame:")
# print(P_W)
if __name__ == "__main__":
    T1 = np.array([
        [8.29042815e-02, -9.96557510e-01, 9.68128146e-05, -2.87201374e-11],
        [9.96557359e-01, 8.29042139e-02, -5.67210941e-04, 9.80477921e-11],
        [5.57232133e-04, 1.43503738e-04, 9.99999834e-01, 1.49213975e-11],
        [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
    ])

    T2 = np.array([
        [8.29042813e-02, -9.96557510e-01, 9.68128146e-05, -4.20477875e-10],
        [9.96557359e-01, 8.29042136e-02, -5.67210941e-04, 1.43566758e-09],
        [5.57232133e-04, 1.43503738e-04, 9.99999834e-01, 2.17468710e-10],
        [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
    ])

    d1 = np.array([11.317639])
    theta1 = np.array([0.45973957]) 
    d2 = np.array([11.317639]) 
    theta2 = np.array([0.45973957])
    print(sonar_triangulation(T1, T2, d1, theta1, d2, theta2))
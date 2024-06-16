import numpy as np

class SonarDataGenerator:
    def __init__(self, P_W, R_SW, t_S, Var_Noise=1.0):
        self.P_W = P_W
        self.R_SW = R_SW
        self.t_S = t_S
        self.Var_Noise = Var_Noise
        self.n = P_W.shape[1]

    def generate_data(self):
        P_S = np.zeros((3, self.n))
        d = np.zeros(self.n)
        cos_theta = np.zeros(self.n)
        sin_theta = np.zeros(self.n)
        tan_theta = np.zeros(self.n)
        theta = np.zeros(self.n)
        cos_phi = np.zeros(self.n)
        P_SI = np.zeros((2, self.n))

        for i in range(self.n):
            P_S[:, i] = self.R_SW @ self.P_W[:, i] + self.t_S
            d[i] = np.linalg.norm(P_S[:, i])
            cos_theta[i] = P_S[0, i] / np.sqrt(P_S[0, i]**2 + P_S[1, i]**2)
            sin_theta[i] = P_S[1, i] / np.sqrt(P_S[0, i]**2 + P_S[1, i]**2)
            tan_theta[i] = sin_theta[i] / cos_theta[i]
            theta[i] = np.arctan(tan_theta[i])
            cos_phi[i] = np.sqrt(P_S[0, i]**2 + P_S[1, i]**2) / d[i]
            P_SI[0, i] = d[i] * cos_theta[i]
            P_SI[1, i] = d[i] * sin_theta[i]
        
        P_SI_Noise = P_SI + self.Var_Noise * np.random.randn(2, self.n)
        return P_S, P_SI, P_SI_Noise

class AnPAlgorithm:
    def __init__(self):
        # t_s, R_sw
        self.R_sw = None
        self.t_s = None

    @staticmethod
    def orthogonalize(r1_Noise, r2_Noise):
        angle_Noise_rad = np.arccos(np.dot(r1_Noise, r2_Noise) / (np.linalg.norm(r1_Noise) * np.linalg.norm(r2_Noise)))
        angle_tran = (np.pi / 2 - angle_Noise_rad) / 2
        k = np.cross(r1_Noise, r2_Noise)
        k = k / np.linalg.norm(k)
        r1_Noise_new = (r1_Noise * np.cos(-angle_tran) + 
                        np.cross(k, r1_Noise) * np.sin(-angle_tran) + 
                        k * np.dot(k, r1_Noise) * (1 - np.cos(-angle_tran)))
        r2_Noise_new = (r2_Noise * np.cos(angle_tran) + 
                        np.cross(k, r2_Noise) * np.sin(angle_tran) + 
                        k * np.dot(k, r2_Noise) * (1 - np.cos(angle_tran)))
        return r1_Noise_new, r2_Noise_new

    @staticmethod
    def rot2aa(R):
        theta = np.arccos((np.trace(R) - 1) / 2)
        if theta == 0:
            k = np.array([0, 0, 0])
        else:
            # if theta < 0.01:
            #     print('Warning: theta is too small, bad conditioned problem')
            k = np.array([(R[2, 1] - R[1, 2]),
                          (R[0, 2] - R[2, 0]),
                          (R[1, 0] - R[0, 1])]) / (2 * np.sin(theta))
        return k, theta

    def compute_t_R(self, P_SI, P_W):
        num = P_SI.shape[1]
        d_Noise = np.zeros(num)
        cos_theta_Noise = np.zeros(num)
        sin_theta_Noise = np.zeros(num)
        tan_theta_Noise = np.zeros(num)
        theta_N = np.zeros(num)

        for i in range(num):
            d_Noise[i] = np.linalg.norm(P_SI[:, i])
            cos_theta_Noise[i] = P_SI[0, i] / d_Noise[i]
            sin_theta_Noise[i] = P_SI[1, i] / d_Noise[i]
            tan_theta_Noise[i] = sin_theta_Noise[i] / cos_theta_Noise[i]
            theta_N[i] = np.arctan(tan_theta_Noise[i])

        count = 0
        Delta_xyz_Noise_my = []
        Delta_d_Noise_my = []

        for i in range(num):
            for j in range(i + 1, num):
                count += 1
                Delta_xyz_Noise_my.append(2 * (P_W[:, j] - P_W[:, i]))
                Delta_d_Noise_my.append(d_Noise[i]**2 - d_Noise[j]**2 - np.linalg.norm(P_W[:, i])**2 + np.linalg.norm(P_W[:, j])**2)

        Delta_xyz_Noise_my = np.array(Delta_xyz_Noise_my)
        Delta_d_Noise_my = np.array(Delta_d_Noise_my).reshape(-1, 1)
        t_W_Noise_my = np.linalg.inv(Delta_xyz_Noise_my.T @ Delta_xyz_Noise_my) @ Delta_xyz_Noise_my.T @ Delta_d_Noise_my

        A_Noise_my = np.zeros((num, 6))

        for i in range(num):
            A_Noise_my[i, 0] = tan_theta_Noise[i] * (P_W[0, i] - t_W_Noise_my[0])
            A_Noise_my[i, 1] = tan_theta_Noise[i] * (P_W[1, i] - t_W_Noise_my[1])
            A_Noise_my[i, 2] = tan_theta_Noise[i] * (P_W[2, i] - t_W_Noise_my[2])
            A_Noise_my[i, 3] = -(P_W[0, i] - t_W_Noise_my[0])
            A_Noise_my[i, 4] = -(P_W[1, i] - t_W_Noise_my[1])
            A_Noise_my[i, 5] = -(P_W[2, i] - t_W_Noise_my[2])

        U_Noise_my, S_Noise_my, V_Noise_my = np.linalg.svd(A_Noise_my)
        r1_Noise_my = np.sqrt(2) * V_Noise_my.T[:3, 5]
        r2_Noise_my = np.sqrt(2) * V_Noise_my.T[3:, 5]

        if abs(np.dot(r1_Noise_my, r2_Noise_my)) <= 1e-4:
            # print('向量 r1_Noise_my 和向量 r2_Noise_my 是正交的。')
            r3_Noise_my = np.cross(r1_Noise_my, r2_Noise_my)
        else:
            # print('向量 r1_Noise_my 和向量 r2_Noise_my 不是正交的。')
            r1_Noise_my, r2_Noise_my = self.orthogonalize(r1_Noise_my, r2_Noise_my)
            # if abs(np.dot(r1_Noise_my, r2_Noise_my)) <= 1e-4:
            #     print('向量 r1_Noise_my_new 和向量 r2_Noise_my_new 是正交的。')
            # else:
            #     print('向量 r1_Noise_my_new 和向量 r2_Noise_my_new 不是正交的。')
            r3_Noise_my = np.cross(r1_Noise_my, r2_Noise_my)
            r1_Noise_my /= np.linalg.norm(r1_Noise_my)
            r2_Noise_my /= np.linalg.norm(r2_Noise_my)
            r3_Noise_my /= np.linalg.norm(r3_Noise_my)

        R_Noise_my_1 = np.vstack([r1_Noise_my, r2_Noise_my, r3_Noise_my])
        R_Noise_my_2 = np.vstack([r1_Noise_my, r2_Noise_my, -r3_Noise_my])
        R_Noise_my_3 = np.vstack([-r1_Noise_my, -r2_Noise_my, r3_Noise_my])
        R_Noise_my_4 = np.vstack([-r1_Noise_my, -r2_Noise_my, -r3_Noise_my])
        
        
        # 根据 R_sw 估计声呐坐标系中的坐标 P_S
        P_S_Estimate_my_1 = R_Noise_my_1 @ (P_W - t_W_Noise_my)
        P_S_Estimate_my_2 = R_Noise_my_2 @ (P_W - t_W_Noise_my)
        P_S_Estimate_my_3 = R_Noise_my_3 @ (P_W - t_W_Noise_my)
        P_S_Estimate_my_4 = R_Noise_my_4 @ (P_W - t_W_Noise_my)

        # 计算估计的 cos(theta)
        cos_theta_vatify_1 = P_S_Estimate_my_1[0, 0] / np.sqrt(P_S_Estimate_my_1[0, 0]**2 + P_S_Estimate_my_1[1, 0]**2)
        cos_theta_vatify_2 = P_S_Estimate_my_2[0, 0] / np.sqrt(P_S_Estimate_my_2[0, 0]**2 + P_S_Estimate_my_2[1, 0]**2)
        cos_theta_vatify_3 = P_S_Estimate_my_3[0, 0] / np.sqrt(P_S_Estimate_my_3[0, 0]**2 + P_S_Estimate_my_3[1, 0]**2)
        cos_theta_vatify_4 = P_S_Estimate_my_4[0, 0] / np.sqrt(P_S_Estimate_my_4[0, 0]**2 + P_S_Estimate_my_4[1, 0]**2)

        # 计算真值 cos(theta)
        cos_theta_true = P_SI[0, 0] / np.sqrt(P_SI[0, 0]**2 + P_SI[1, 0]**2)

        # 选择最优的 R_sw
        if cos_theta_vatify_1 * cos_theta_true > 0:
            R_sw = R_Noise_my_1
        elif cos_theta_vatify_2 * cos_theta_true > 0:
            R_sw = R_Noise_my_2
        elif cos_theta_vatify_3 * cos_theta_true > 0:
            R_sw = R_Noise_my_3
        elif cos_theta_vatify_4 * cos_theta_true > 0:
            R_sw = R_Noise_my_4
        else:
            raise ValueError("No valid R_sw found")

        t_s = -R_sw @ t_W_Noise_my
        self.R_sw = R_sw
        self.t_s = t_s
        
        return t_s, R_sw

    def estimate_accuracy(self, R_sw_gt):
        k, theta = self.rot2aa(R_sw_gt.T @ self.R_sw)
        return k, theta

if __name__ == "__main__":
    # 初始化参数
    P_W = np.array([[30, 41, 21, 13, 23, 73, 35, 66, 72, 82, 15],
                    [44, 26, 63, 34, 15, 22, 14, 33, 25, 23, 42],
                    [35, 17, 16, 57, 54, 61, 42, 11, 13, 3, 47]])
    R_SW = np.array([[-0.5798, 0.4836, -0.6557],
                    [-0.8135, -0.3883, 0.4329],
                    [-0.0453, 0.7844, 0.6186]])
    t_S = np.array([6, 4, 7])

    # 实例化数据生成器
    data_generator = SonarDataGenerator(P_W, R_SW, t_S, Var_Noise=0.1)

    # 生成数据
    P_S, P_SI, P_SI_Noise = data_generator.generate_data()

    # 实例化算法
    anp_algorithm = AnPAlgorithm()

    # 计算 t_s 和 R_SW_Noise_my
    t_s_cal, R_sw_cal = anp_algorithm.compute_t_R(P_SI_Noise, P_W)

    print("t_s_cal: \n", t_s_cal)
    print("R_sw_cal: \n", R_sw_cal)

    # 估计精度
    k, theta = anp_algorithm.estimate_accuracy(R_SW)

    print("估计的精度 theta:", theta)
        
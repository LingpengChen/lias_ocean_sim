%% 加载和处理rosbag数据
clear all;
close all;
clc;

% 加载.mat文件
data = load('./rss/rss_data.mat');

%% 提取时间序列数据
% Ground Truth位姿数据
t_pose_gt = data.pose_gt.time;
position_gt = data.pose_gt.position;
orientation_gt = data.pose_gt.orientation;
twist_linear_gt = data.pose_gt.twist_linear;
twist_angular_gt = data.pose_gt.twist_angular;

% IMU数据
t_imu = data.imu.time;
imu_angular_vel = data.imu.angular_velocity;
imu_linear_acc = data.imu.linear_acceleration;
imu_orientation = data.imu.orientation;

% DVL数据
t_dvl = data.dvl.time;
dvl_velocity = data.dvl.twist_linear;
dvl_covariance = data.dvl.covariance;

% Charuco相机位姿数据
t_charuco = data.charuco.time;
charuco_position = data.charuco.position;
charuco_orientation = data.charuco.orientation;

% 声呐数据
t_sonar = data.sonar.time;
sonar_distances = data.sonar.distances_gt;
sonar_thetas = data.sonar.theta_gt;
sonar_pose = data.sonar.pose;

%% 绘图

% 创建图形窗口
figure('Name', 'Trajectory and Sensor Data', 'Position', [100, 100, 1200, 800]);

% 1. Ground Truth轨迹
subplot(2,3,1)
plot3(position_gt(:,1), position_gt(:,2), position_gt(:,3), 'b-', 'LineWidth', 2);
grid on;
xlabel('X (m)');
ylabel('Y (m)');
zlabel('Z (m)');
title('Ground Truth Trajectory');
axis equal;

% 2. IMU数据
subplot(2,3,2)
plot(t_imu, imu_angular_vel);
grid on;
xlabel('Time (s)');
ylabel('Angular Velocity (rad/s)');
title('IMU Angular Velocity');
legend('x', 'y', 'z');

% 3. DVL速度
subplot(2,3,3)
plot(t_dvl, dvl_velocity);
grid on;
xlabel('Time (s)');
ylabel('Velocity (m/s)');
title('DVL Velocity');
legend('v_x', 'v_y', 'v_z');

% 4. Charuco相机位置
subplot(2,3,4)
plot(t_charuco, charuco_position);
grid on;
xlabel('Time (s)');
ylabel('Position (m)');
title('Charuco Camera Position');
legend('x', 'y', 'z');

% 5. 声呐数据
subplot(2,3,5)
% 创建热图
imagesc(sonar_distances');
colorbar;
xlabel('Measurement Index');
ylabel('Beam Index');
title('Sonar Distances');

% 6. 速度对比
subplot(2,3,6)
hold on;
plot(t_pose_gt, vecnorm(twist_linear_gt, 2, 2), 'b-', 'LineWidth', 1.5);
plot(t_dvl, vecnorm(dvl_velocity, 2, 2), 'r--', 'LineWidth', 1.5);
grid on;
xlabel('Time (s)');
ylabel('Speed (m/s)');
title('Speed Comparison');
legend('Ground Truth', 'DVL');
hold off;

%% 数据分析
% 计算采样率
fs_pose_gt = 1/mean(diff(t_pose_gt));
fs_imu = 1/mean(diff(t_imu));
fs_dvl = 1/mean(diff(t_dvl));
fs_charuco = 1/mean(diff(t_charuco));
fs_sonar = 1/mean(diff(t_sonar));

% 打印基本信息
fprintf('数据统计信息:\n');
fprintf('数据时长: %.2f 秒\n', t_pose_gt(end) - t_pose_gt(1));
fprintf('采样率:\n');
fprintf('- Ground Truth: %.2f Hz\n', fs_pose_gt);
fprintf('- IMU: %.2f Hz\n', fs_imu);
fprintf('- DVL: %.2f Hz\n', fs_dvl);
fprintf('- Charuco: %.2f Hz\n', fs_charuco);
fprintf('- Sonar: %.2f Hz\n', fs_sonar);

% 计算轨迹总长度
total_distance = sum(sqrt(sum(diff(position_gt).^2, 2)));
fprintf('轨迹总长度: %.2f 米\n', total_distance);

%% 可选：将四元数转换为欧拉角
% 需要Robotics System Toolbox
if exist('quat2eul', 'file')
    euler_gt = quat2eul([orientation_gt(:,4), orientation_gt(:,1:3)]);
    euler_charuco = quat2eul([charuco_orientation(:,4), charuco_orientation(:,1:3)]);
    
    figure('Name', 'Orientation Analysis');
    subplot(2,1,1)
    plot(t_pose_gt, rad2deg(euler_gt));
    grid on;
    xlabel('Time (s)');
    ylabel('Angle (deg)');
    title('Ground Truth Orientation');
    legend('Roll', 'Pitch', 'Yaw');
    
    subplot(2,1,2)
    plot(t_charuco, rad2deg(euler_charuco));
    grid on;
    xlabel('Time (s)');
    ylabel('Angle (deg)');
    title('Charuco Camera Orientation');
    legend('Roll', 'Pitch', 'Yaw');
end

%% 保存图形
saveas(gcf, 'sensor_data_analysis.fig');
saveas(gcf, 'sensor_data_analysis.png');
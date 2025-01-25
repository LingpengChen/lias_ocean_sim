#!/usr/bin/env python3
import rosbag
import scipy.io as sio
import numpy as np

def bag_to_mat(bag_file, output_file):
    # 创建数据字典
    data = {
        'pose_gt': {'time': [], 'position': [], 'orientation': [], 'twist_linear': [], 'twist_angular': []},
        'imu': {'time': [], 'angular_velocity': [], 'linear_acceleration': [], 'orientation': []},
        'dvl': {'time': [], 'twist_linear': [], 'covariance': []},
        'charuco': {'time': [], 'position': [], 'orientation': []},
        'sonar': {'time': [], 'distances_gt': [], 'theta_gt': [], 'pose': []}
    }
    
    # 读取bag文件
    bag = rosbag.Bag(bag_file)
    
    # 遍历每个话题
    for topic, msg, t in bag.read_messages():
        time = t.to_sec()
        
        if topic == '/rexrov/pose_gt':  # nav_msgs/Odometry
            data['pose_gt']['time'].append(time)
            # Position
            pos = [msg.pose.pose.position.x, 
                  msg.pose.pose.position.y, 
                  msg.pose.pose.position.z]
            data['pose_gt']['position'].append(pos)
            # Orientation
            orient = [msg.pose.pose.orientation.x,
                     msg.pose.pose.orientation.y,
                     msg.pose.pose.orientation.z,
                     msg.pose.pose.orientation.w]
            data['pose_gt']['orientation'].append(orient)
            # Twist
            twist_linear = [msg.twist.twist.linear.x,
                          msg.twist.twist.linear.y,
                          msg.twist.twist.linear.z]
            twist_angular = [msg.twist.twist.angular.x,
                           msg.twist.twist.angular.y,
                           msg.twist.twist.angular.z]
            data['pose_gt']['twist_linear'].append(twist_linear)
            data['pose_gt']['twist_angular'].append(twist_angular)
            
        elif topic == '/rexrov/imu':  # sensor_msgs/Imu
            data['imu']['time'].append(time)
            # Angular velocity
            ang_vel = [msg.angular_velocity.x,
                      msg.angular_velocity.y,
                      msg.angular_velocity.z]
            data['imu']['angular_velocity'].append(ang_vel)
            # Linear acceleration
            lin_acc = [msg.linear_acceleration.x,
                      msg.linear_acceleration.y,
                      msg.linear_acceleration.z]
            data['imu']['linear_acceleration'].append(lin_acc)
            # Orientation
            orient = [msg.orientation.x,
                     msg.orientation.y,
                     msg.orientation.z,
                     msg.orientation.w]
            data['imu']['orientation'].append(orient)
            
        elif topic == '/rexrov/dvl_twist':  # geometry_msgs/TwistWithCovarianceStamped
            data['dvl']['time'].append(time)
            # Linear velocity
            twist = [msg.twist.twist.linear.x,
                    msg.twist.twist.linear.y,
                    msg.twist.twist.linear.z]
            data['dvl']['twist_linear'].append(twist)
            # Covariance (6x6 matrix stored as 36-element array)
            data['dvl']['covariance'].append(list(msg.twist.covariance))
            
        elif topic == '/charuco_pose':  # geometry_msgs/PoseStamped
            data['charuco']['time'].append(time)
            # Position
            pos = [msg.pose.position.x,
                  msg.pose.position.y,
                  msg.pose.position.z]
            data['charuco']['position'].append(pos)
            # Orientation
            orient = [msg.pose.orientation.x,
                     msg.pose.orientation.y,
                     msg.pose.orientation.z,
                     msg.pose.orientation.w]
            data['charuco']['orientation'].append(orient)
            
        elif topic == '/preprocessed_sonar_data':  # SonarData
            data['sonar']['time'].append(time)
            data['sonar']['distances_gt'].append(list(msg.distances_gt))
            data['sonar']['theta_gt'].append(list(msg.theta_gt))
            data['sonar']['pose'].append(list(msg.pose))
    
    # 转换列表为numpy数组
    for key in data:
        for subkey in data[key]:
            data[key][subkey] = np.array(data[key][subkey])
    
    # 保存为.mat文件
    sio.savemat(output_file, data)
    bag.close()
    print(f"数据已保存到 {output_file}")

if __name__ == '__main__':
    bag_file = './rss/rss_2025-01-25-11-51-45.bag'
    output_file = './rss/rss_data.mat'
    bag_to_mat(bag_file, output_file)
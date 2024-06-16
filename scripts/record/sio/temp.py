import rospy
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped

class TrajectoryPublisher:
    def __init__(self):
        # 初始化ROS节点
        rospy.init_node('trajectory_publisher')

        # 初始化Path发布者
        self.path_pub = rospy.Publisher('trajectory_path', Path, queue_size=10)

        # 初始化trajectory，这里假设你已经填充了trajectory
        self.trajectory = {
            'poses': [
                {'position': {'x': 0.0, 'y': 0.0, 'z': 0.0}, 'orientation': {'x': 0.0, 'y': 0.0, 'z': 0.0, 'w': 1.0}},
                {'position': {'x': 1.0, 'y': 0.0, 'z': 0.0}, 'orientation': {'x': 0.0, 'y': 0.0, 'z': 0.0, 'w': 1.0}},
                {'position': {'x': 2.0, 'y': 1.0, 'z': 0.0}, 'orientation': {'x': 0.0, 'y': 0.0, 'z': 0.0, 'w': 1.0}},
                {'position': {'x': 3.0, 'y': 1.0, 'z': 0.0}, 'orientation': {'x': 0.0, 'y': 0.0, 'z': 0.0, 'w': 1.0}},
            ]
        }
    
    def publish_trajectory(self):
        path_msg = Path()
        path_msg.header.stamp = rospy.Time.now()
        path_msg.header.frame_id = "map"

        for pose in self.trajectory['poses']:
            pose_stamped = PoseStamped()
            pose_stamped.header.stamp = rospy.Time.now()
            pose_stamped.header.frame_id = "map"
            pose_stamped.pose.position.x = pose['position']['x']
            pose_stamped.pose.position.y = pose['position']['y']
            pose_stamped.pose.position.z = pose['position']['z']
            pose_stamped.pose.orientation.x = pose['orientation']['x']
            pose_stamped.pose.orientation.y = pose['orientation']['y']
            pose_stamped.pose.orientation.z = pose['orientation']['z']
            pose_stamped.pose.orientation.w = pose['orientation']['w']

            path_msg.poses.append(pose_stamped)

        self.path_pub.publish(path_msg)

if __name__ == '__main__':
    try:
        # 初始化发布类
        traj_publisher = TrajectoryPublisher()

        # 循环发布轨迹
        rate = rospy.Rate(1)  # 1 Hz
        while not rospy.is_shutdown():
            traj_publisher.publish_trajectory()
            rate.sleep()
    except rospy.ROSInterruptException:
        pass

#!/usr/bin/python3
import subprocess
import signal
import time

def record_bag(topics, duration=None, output_path='./my_rosbags/session'):
    """
    Record a rosbag with the specified topics and save it to a specified path.
    
    Args:
    topics (list): List of topic names to record.
    duration (float, optional): Duration in seconds to record the bag. If None, the recording
                                continues until the script is manually stopped.
    output_path (str): The file path where the rosbag will be saved.
    """
    # Construct the rosbag record command with output path
    cmd = ['rosbag', 'record', '-o', output_path] + topics

    # Start recording
    process = subprocess.Popen(cmd)
    print("Recording started...")

    try:
        if duration is not None:
            print("Recording will stop after {} seconds.".format(duration))
            # Wait for the specified duration
            time.sleep(duration)
            # Terminate the process
            process.send_signal(signal.SIGINT)
            print("DONE")
        else:
            # Wait indefinitely until interrupted
            process.wait()
    except KeyboardInterrupt:
        # Handle Ctrl-C to stop recording
        process.send_signal(signal.SIGINT)
        print("Recording manually stopped.")
    finally:
        process.wait()  # Ensure recording is stopped properly

if __name__ == '__main__':
    # Define the topics to record
    topics = [
        '/rexrov/pose_gt',  # pose ground truth nav_msgs/Odometry
        '/rexrov/imu', # sensor_msgs/Imu
        '/rexrov/dvl_twist', # dvl geometry_msgs/TwistWithCovarianceStamped
        '/charuco_pose',   # camera  geometry_msgs/PoseStamped
        '/preprocessed_sonar_data' # sonar SonarData
    ]

    # Specify the output path and name
    # output_path = './record/vio/vio'
    output_path = './record/rss/rss'

    # Start recording, change the duration as needed
    record_bag(topics, duration=60, output_path=output_path)  # Record for 3 seconds

#!/usr/bin/python3
from sensor_msgs.msg import LaserScan
import rospy
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
import numpy as np
from sklearn.linear_model import RANSACRegressor

class RANSACLineFitter:
    def __init__(self):
        rospy.init_node('ransac_line_fitter')
        self.scan_sub = rospy.Subscriber('/car_1/scan', LaserScan, self.scan_callback)
        self.marker_pub = rospy.Publisher('/ransac_marker', Marker, queue_size=1)
        
        # Load parameters from the parameter server
        self.num_of_itr = rospy.get_param('~num_of_itr', 100)
        self.inlier_thresh = rospy.get_param('~inlier_thresh', 0.05)
        self.stop_thresh = rospy.get_param('~stop_thresh', 0.95)
        
    def fit_line(self, points):
        if len(points) < 2:
            return None  # Return None if there are not enough points to fit a line
        ransac = RANSACRegressor(min_samples=min(2, len(points)), residual_threshold=self.inlier_thresh, max_trials=self.num_of_itr)
        X = np.array(points)[:, 0].reshape(-1, 1)
        y = np.array(points)[:, 1]
        try:
            ransac.fit(X, y)
            line = ransac.predict(X)
            return line
        except ValueError as e:
            rospy.logerr("RANSAC fitting failed: {}".format(e))
            return None
        
    def scan_callback(self, scan_msg):
        # Extract relevant data points from scan message
        angles = np.arange(scan_msg.angle_min, scan_msg.angle_max, scan_msg.angle_increment)
        ranges = np.array(scan_msg.ranges)
        points = []
        for i, angle in enumerate(angles):
            if not np.isinf(ranges[i]):
                x = ranges[i] * np.cos(angle)
                y = ranges[i] * np.sin(angle)
                points.append((x, y))

        # Apply RANSAC algorithm
        remaining_points = points[:]
        consensus_set = []
        while len(remaining_points) > 0 and len(consensus_set) / len(points) < self.stop_thresh:
            # Fit a line to the remaining points
            line = self.fit_line(remaining_points)

            # Check if line is not None
            if line is not None:
                # Find inliers
                inliers = []
                for point in remaining_points:
                    predicted_value = line[0]  # Get the single predicted value
                    if np.abs(point[1] - predicted_value) < self.inlier_thresh:
                        inliers.append(point)

                consensus_set.extend(inliers)
                remaining_points = [point for point in remaining_points if point not in inliers]
            else:
                break  # Exit the loop if there are not enough points to fit a line

        # Fit final line using all consensus points
        if len(consensus_set) >= 2:  # Ensure there are enough points to fit a line
            final_line = self.fit_line(consensus_set)

            # Publish line marker
            marker_msg = Marker()
            marker_msg.header.frame_id = 'base_link'
            marker_msg.type = Marker.LINE_STRIP
            marker_msg.action = Marker.ADD
            marker_msg.scale.x = 0.1
            marker_msg.color.r = 1.0
            marker_msg.color.a = 1.0
            for point in consensus_set:
                marker_msg.points.append(Point(point[0], point[1], 0))
            self.marker_pub.publish(marker_msg)


if __name__ == '__main__':
    try:
        ransac_line_fitter = RANSACLineFitter()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

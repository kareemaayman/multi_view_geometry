#!/usr/bin/env python3

import rospy
from std_msgs.msg import String
from std_msgs.msg import Header

# Custom messages
from multi_view_geometry.msg import GeometricInliers
from multi_view_geometry.msg import CameraMotion   # ✅ your required message


class MotionEstimationNode:

    def __init__(self):
        rospy.init_node('motion_node', anonymous=True)

        # Parameter (required)
        self.focal_length = rospy.get_param('~focal_length', 525.0)

        # Subscriber
        rospy.Subscriber('/geometric_inliers', GeometricInliers, self.callback)

        # Publishers
        self.motion_pub = rospy.Publisher('/camera_motion', CameraMotion, queue_size=10)
        self.state_pub = rospy.Publisher('/system_state', String, queue_size=10)

        rospy.loginfo("Motion Estimation Node Started")


    def callback(self, msg):

        n = len(msg.query_x)

        # ⚠️ FAILURE CONDITIONS (system rules)
        if n < 20:
            self.publish_failure("LOW_FEATURES")
            return

        dx_total = 0.0
        dy_total = 0.0

        for i in range(n):

            x1 = msg.query_x[i]
            y1 = msg.query_y[i]

            x2 = msg.train_x[i]
            y2 = msg.train_y[i]

            dx_total += (x2 - x1)
            dy_total += (y2 - y1)

        # Average optical flow
        avg_dx = dx_total / n
        avg_dy = dy_total / n

        magnitude = (avg_dx**2 + avg_dy**2) ** 0.5

        # Direction estimation
        dir_h, dir_d = self.estimate_direction(avg_dx, avg_dy)

        # Always monocular
        scale_ambiguous = True

        # Build message
        motion_msg = CameraMotion()

        motion_msg.header = Header()
        motion_msg.header.stamp = rospy.Time.now()

        motion_msg.direction_horizontal = dir_h
        motion_msg.direction_depth = dir_d

        motion_msg.translation_x = avg_dx
        motion_msg.translation_y = avg_dy
        motion_msg.magnitude = magnitude
        motion_msg.scale_ambiguous = scale_ambiguous

        # Publish
        self.motion_pub.publish(motion_msg)

        rospy.loginfo(
            f"[Motion] dx={avg_dx:.2f}, dy={avg_dy:.2f}, mag={magnitude:.2f}, "
            f"H={dir_h}, D={dir_d}"
        )


    def estimate_direction(self, dx, dy):

        threshold = 1.0

        # Horizontal motion
        if abs(dx) > threshold:
            if dx > 0:
                direction_horizontal = "RIGHT"
            else:
                direction_horizontal = "LEFT"
        else:
            direction_horizontal = "NONE"

        # Depth motion (approx from vertical flow)
        if abs(dy) > threshold:
            if dy < 0:
                direction_depth = "FORWARD"
            else:
                direction_depth = "BACKWARD"
        else:
            direction_depth = "NONE"

        return direction_horizontal, direction_depth


    def publish_failure(self, reason):

        rospy.logwarn(f"[Motion] FAILURE: {reason}")

        # system state
        self.state_pub.publish("UNRELIABLE")

        # still publish motion message but marked unknown
        motion_msg = CameraMotion()
        motion_msg.header = Header()
        motion_msg.header.stamp = rospy.Time.now()

        motion_msg.direction_horizontal = "UNKNOWN"
        motion_msg.direction_depth = "UNKNOWN"
        motion_msg.translation_x = 0.0
        motion_msg.translation_y = 0.0
        motion_msg.magnitude = 0.0
        motion_msg.scale_ambiguous = True

        self.motion_pub.publish(motion_msg)


if __name__ == '__main__':
    try:
        MotionEstimationNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

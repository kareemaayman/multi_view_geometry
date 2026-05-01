#!/usr/bin/env python3
"""
=============================================================
 Node 1 — Camera Stream Node  (Student 5)
=============================================================
Responsibility:
  Capture frames from webcam or video file and publish them
  as ROS sensor_msgs/Image messages.

Topics Published:
  /camera_frames  (sensor_msgs/Image)

Parameters:
  ~camera_source  : int (webcam index) or str (video path) [default: 0]
  ~frame_rate     : publishing rate in Hz                  [default: 10]

ROS1 vs ROS2 differences:
  ROS1: cv_bridge.CvBridge().cv2_to_imgmsg()
  ROS2: cv_bridge.CvBridge().cv2_to_imgmsg() — same API but uses
        rclpy instead of rospy and different node init pattern.
=============================================================
"""
import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class CameraStreamNode:
    def __init__(self):
        rospy.init_node('camera_node', anonymous=False)

        # ── Parameters ──────────────────────────────────────────────────
        self.camera_source = rospy.get_param('~camera_source', 0)
        self.frame_rate    = rospy.get_param('~frame_rate', 10)

        # Convert string "0" → int 0 (for webcam index passed via launch file)
        try:
            self.camera_source = int(self.camera_source)
        except (ValueError, TypeError):
            pass  # keep as string — it's a video file path

        # ── ROS Communication ────────────────────────────────────────────
        self.bridge = CvBridge()
        self.pub    = rospy.Publisher('/camera_frames', Image, queue_size=10)

        # ── OpenCV Capture ───────────────────────────────────────────────
        self.cap = cv2.VideoCapture(self.camera_source)
        if not self.cap.isOpened():
            rospy.logerr("[Camera] Cannot open source: %s", str(self.camera_source))
            rospy.signal_shutdown("Camera source unavailable")
            return

        rospy.loginfo("[Camera] Started | source=%s | rate=%d FPS",
                      str(self.camera_source), self.frame_rate)

    def run(self):
        rate = rospy.Rate(self.frame_rate)
        while not rospy.is_shutdown():
            ret, frame = self.cap.read()

            if not ret:
                # Loop video file; for webcam this means a real error
                rospy.logwarn("[Camera] Frame grab failed — looping or retrying")
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            # ── Convert & Publish ────────────────────────────────────────
            msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
            msg.header.stamp    = rospy.Time.now()
            msg.header.frame_id = 'camera_frame'
            self.pub.publish(msg)

            rate.sleep()

        self.cap.release()
        rospy.loginfo("[Camera] Shutting down, camera released.")


if __name__ == '__main__':
    try:
        node = CameraStreamNode()
        node.run()
    except rospy.ROSInterruptException:
        pass

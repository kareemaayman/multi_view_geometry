#!/usr/bin/env python3
"""
=============================================================
 Node 4 — Feature Matching Node  (Student 3)
=============================================================
Responsibility:
  Match ORB descriptors between consecutive frames using a
  Brute-Force matcher with the Hamming distance metric.
  Applies Lowe's ratio test (k=2 nearest neighbours) to
  discard ambiguous matches before publishing.

Topics Subscribed:
  /descriptors  (multi_view_geometry/DescriptorArray)

Topics Published:
  /raw_matches  (multi_view_geometry/MatchArray)

Parameters:
  ~match_threshold : Lowe ratio test threshold [default: 0.75]

Internal Logic:
  1. Buffer the previous DescriptorArray
  2. On each new DescriptorArray: knnMatch(prev, curr, k=2)
  3. Keep match m if  m.distance < threshold * second_best.distance
  4. Embed pixel coordinates from both frames into MatchArray

=============================================================
"""
import rospy
import cv2
import numpy as np
from multi_view_geometry.msg import DescriptorArray, MatchArray


class FeatureMatchingNode:
    def __init__(self):
        rospy.init_node('matching_node', anonymous=False)

        self.threshold = rospy.get_param('~match_threshold', 0.75)

        # BFMatcher with Hamming distance (required for binary ORB descriptors)
        self.matcher  = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self.prev     = None   # (arr, xs, ys) of previous frame

        self.pub = rospy.Publisher('/raw_matches', MatchArray, queue_size=10)
        rospy.Subscriber('/descriptors', DescriptorArray, self.callback, queue_size=5)

        rospy.loginfo("[Matching] Started | ratio_threshold=%.2f", self.threshold)
        rospy.spin()

    def callback(self, desc_msg):
        if desc_msg.num_keypoints < 2:
            rospy.logwarn("[Matching] Too few descriptors (%d) to match", desc_msg.num_keypoints)
            return

        # Reconstruct uint8 numpy descriptor matrix
        curr_arr = np.array(desc_msg.data, dtype=np.uint8).reshape(
            desc_msg.num_keypoints, desc_msg.descriptor_size)
        curr_xs  = list(desc_msg.x)
        curr_ys  = list(desc_msg.y)

        # ── First frame: just store, nothing to match yet ─────────────────
        if self.prev is None:
            self.prev = (curr_arr, curr_xs, curr_ys)
            rospy.loginfo("[Matching] First frame cached — waiting for next frame")
            return

        prev_arr, prev_xs, prev_ys = self.prev

        # ── kNN matching (k=2 for ratio test) ────────────────────────────
        try:
            knn = self.matcher.knnMatch(prev_arr, curr_arr, k=2)
        except cv2.error as e:
            rospy.logwarn("[Matching] knnMatch error: %s", str(e))
            self.prev = (curr_arr, curr_xs, curr_ys)
            return

        # ── Lowe's ratio test ─────────────────────────────────────────────
        msg        = MatchArray()
        msg.header = desc_msg.header

        for pair in knn:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < self.threshold * n.distance:
                msg.query_idx.append(m.queryIdx)
                msg.train_idx.append(m.trainIdx)
                msg.query_x.append(float(prev_xs[m.queryIdx]))
                msg.query_y.append(float(prev_ys[m.queryIdx]))
                msg.train_x.append(float(curr_xs[m.trainIdx]))
                msg.train_y.append(float(curr_ys[m.trainIdx]))
                msg.distance.append(float(m.distance))

        msg.count = len(msg.query_idx)

        # Update previous frame buffer
        self.prev = (curr_arr, curr_xs, curr_ys)

        self.pub.publish(msg)
        rospy.loginfo("[Matching] %d raw matches (from %d vs %d keypoints)",
                      msg.count, len(prev_xs), len(curr_xs))


if __name__ == '__main__':
    FeatureMatchingNode()

#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the

import rospy
import tf
from geometry_msgs.msg import PoseStamped

import sys
sys.path.append('/home/autolabor/home-robot/src/home_robot') #TODO change the path


from home_robot.utils.pose import to_matrix
from home_robot_hw.ros.utils import matrix_to_pose_msg


class CameraPosePublisher(object):
    """publishes the camera pose constantly so that we do not have a dependency on /tf"""

    def __init__(self, topic_name: str = "camera_pose"):
        self._pub = rospy.Publisher(topic_name, PoseStamped, queue_size=10)
        self._listener = tf.TransformListener()
        self._seq = 0
        # print("hi")

    def spin(self, rate=10):
        # print(">>")
        rate = rospy.Rate(rate)
        while not rospy.is_shutdown():
            try:
                (trans, rot) = self._listener.lookupTransform(
                    "map", "camera_link", rospy.Time(0)
                )
                matrix = to_matrix(trans, rot)
                # print("helo")s
                msg = PoseStamped(pose=matrix_to_pose_msg(matrix))
                msg.header.stamp = rospy.Time.now()
                msg.header.seq = self._seq
                self._pub.publish(msg)
                self._seq += 1
            except (
                tf.LookupException,
                tf.ConnectivityException,
                tf.ExtrapolationException,
            ):
                # print("what")
                continue
        rate.sleep()


if __name__ == "__main__":
    rospy.init_node("camera_pose_publisher")
    publisher = CameraPosePublisher()
    publisher.spin()

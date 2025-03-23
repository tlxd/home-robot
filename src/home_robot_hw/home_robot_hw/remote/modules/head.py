# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import rclpy

class StretchHeadClient():

    def __init__(
        self,
        ros_client
    ):
        super().__init__()

        self._ros_client = ros_client
        # self._wait_for_pose()

    # Interface methods

    def get_pose(self):
        """get matrix version of the camera pose"""
        mat = self._ros_client.se3_camera_pose.matrix()
        return mat

    def _wait_for_pose(self):
        """wait until we have an accurate pose estimate"""
        tmp_node = rclpy.create_node('wait_for_pose_node')  # 创建一个临时节点
        rate = tmp_node.create_rate(10)  # 使用临时节点创建速率对象
        while rclpy.ok():
            if self._ros_client.se3_camera_pose is not None:
                break
            rate.sleep()
        tmp_node.destroy_node()  # 销毁临时节点

    def get_images(self, compute_xyz=True):
        """helper logic to get images from the robot's camera feed"""
        rgb = self._ros_client.rgb_cam.get()
        if self._ros_client.filter_depth:
            dpt = self._ros_client.dpt_cam.get_filtered()
        else:
            dpt = self._ros_client.dpt_cam.get()

        # Compute point cloud from depth image
        if compute_xyz:
            xyz = self._ros_client.dpt_cam.depth_to_xyz(
                self._ros_client.dpt_cam.fix_depth(dpt)
            )
            imgs = [rgb, dpt, xyz]
        else:
            imgs = [rgb, dpt]
            xyz = None

        return imgs

    def shutdown(self):
        """Clean up resources"""
        self._tf_listener = None
        self._tf_buffer = None
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import threading
from collections import deque

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.task import Future
from sensor_msgs.msg import CameraInfo, Image

from home_robot.utils.image import Camera
from home_robot_hw.ros.msg_numpy import image_to_numpy


class RosCamera(Camera):
    """compute camera parameters from ROS instead"""

    def __init__(
        self,
        node: Node,
        name: str = "/camera/color",
        verbose: bool = False,
        rotations: int = 0,
        buffer_size: int = None,
    ):
        """
        Args:
            name: Image topic name
            verbose: Whether or not to print out camera info
            rotations: Number of counterclockwise rotations for the output image array
            buffer_size: Size of buffer for intialization and filtering
        """
        self.node = node
        self.name = name
        self.rotations = 0

        # Initialize
        self._img = None
        self._t = self.node.get_clock().now()
        self._lock = threading.Lock()
        self._camera_info_topic = "/camera/aligned_depth_to_color/camera_info"

        # if verbose:
        #     print("Waiting for camera info on", self._camera_info_topic + "...")

        # 创建一个 Future 对象用于等待相机信息
        cam_info_future = Future()

        def cam_info_callback(msg):
            nonlocal cam_info_future
            if not cam_info_future.done():
                cam_info_future.set_result(msg)

        # 创建相机信息的订阅者
        cam_info_sub = self.node.create_subscription(
            CameraInfo,
            self._camera_info_topic,
            cam_info_callback,
            10
        )

        # 阻塞等待相机信息
        while rclpy.ok() and not cam_info_future.done():
            rclpy.spin_once(self.node, timeout_sec=0.1)

        # 销毁订阅者
        cam_info_sub.destroy()

        # 获取相机信息
        cam_info = cam_info_future.result()

        # Buffer
        self.buffer_size = buffer_size
        if self.buffer_size is not None:
            # create buffer
            self._buffer = deque()
        self.height = cam_info.height
        self.width = cam_info.width
        self.pos, self.orn, self.pose_matrix = None, None, None

        # Get camera information
        self.distortion_model = cam_info.distortion_model
        self.D = np.array(cam_info.D)  # Distortion parameters
        self.K = np.array(cam_info.K).reshape(3, 3)
        self.R = np.array(cam_info.R).reshape(3, 3)  # Rectification matrix
        self.P = np.array(cam_info.P).reshape(3, 4)  # Projection/camera matrix

        self.fx = self.K[0, 0]
        self.fy = self.K[1, 1]
        self.px = self.K[0, 2]
        self.py = self.K[1, 2]

        self.near_val = 0.1
        self.far_val = 7.0
        self.frame_id = self.cam_info.header.frame_id
        self.topic_name = name
        self._subscriber = self.node.create_subscription(
            Image,
            self.topic_name,
            self._cb,
            1
        )

    def _cb(self, msg):
        """capture the latest image and save it"""
        with self._lock:
            img = image_to_numpy(msg)

            # Preprocess encoding
            if msg.encoding == "16UC1":
                # depth support goes here
                # Convert the image to metric (meters)
                img = img / 1000.0
            elif msg.encoding == "rgb8":
                # color support - do nothing
                pass

            # Image orientation
            self._img = np.rot90(img, k=0)

            # Add to buffer
            self._t = msg.header.stamp
            if self.buffer_size is not None:
                self._add_to_buffer(img)

    def _add_to_buffer(self, img):
        """add to buffer and remove old image if buffer size exceeded"""
        self._buffer.append(img)
        if len(self._buffer) > self.buffer_size:
            self._buffer.popleft()

    def valid_mask(self, depth):
        """return only valid pixels"""
        depth = depth.reshape(-1)
        return np.bitwise_and(depth > self.near_val, depth < self.far_val)

    def valid_pc(self, xyz, rgb, depth):
        mask = self.valid_mask(depth)
        xyz = xyz.reshape(-1, 3)[mask]
        rgb = rgb.reshape(-1, 3)[mask]
        return xyz, rgb

    def get_time(self):
        """Get time image was received last"""
        return self._t

    def wait_for_image(self) -> None:
        """Wait for image. Needs to be sort of slow, in order to make sure we give it time
        to update the image in the backend."""
        self.node.get_clock().sleep_for(rclpy.time.Duration(seconds=0.2))
        rate = self.node.create_rate(5)
        while rclpy.ok():
            with self._lock:
                if self.buffer_size is None:
                    if self._img is not None:
                        break
                else:
                    # Wait until we have a full buffer
                    if len(self._buffer) >= self.buffer_size:
                        break
            rate.sleep()

    def get(self, device=None):
        """return the current image associated with this camera"""
        with self._lock:
            if self._img is None:
                return None
            else:
                # Return a copy
                img = self._img.copy()

        if device is not None:
            # If a device is specified, assume we want to move to pytorch
            import torch

            img = torch.FloatTensor(img).to(device)

        return img

    def get_filtered(self, std_threshold=0.005, device=None):
        """get image from buffer; do some smoothing"""
        if self.buffer_size is None:
            raise RuntimeError("no buffer")
        with self._lock:
            imgs = [img[None] for img in self._buffer]
        # median = np.median(np.concatenate(imgs, axis=0), axis=0)
        stacked = np.concatenate(imgs, axis=0)
        avg = np.mean(stacked, axis=0)
        std = np.std(stacked, axis=0)
        dims = avg.shape
        avg = avg.reshape(-1)
        avg[std.reshape(-1) > std_threshold] = 0
        img = avg.reshape(*dims)

        if device is not None:
            # If a device is specified, assume we want to move to pytorch
            import torch

            img = torch.FloatTensor(img).to(device)

        return img

    def get_frame(self):
        return self.frame_id

    def get_K(self):
        return self.K.copy()

    def get_info(self):
        return {
            "D": self.D,
            "K": self.K,
            "fx": self.fx,
            "fy": self.fy,
            "px": self.px,
            "py": self.py,
            "near_val": self.near_val,
            "far_val": self.far_val,
            "R": self.R,
            "P": self.P,
            "height": self.height,
            "width": self.width,
        }
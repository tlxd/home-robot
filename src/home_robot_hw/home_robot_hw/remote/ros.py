# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import  Optional


import ros_numpy
import rclpy
from rclpy.node import Node
from rclpy.time import Time, Duration
from rclpy.client import Client
import sophuspy as sp
import tf2_ros
from geometry_msgs.msg import  Pose, PoseStamped, Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool
from std_srvs.srv import SetBool, Trigger

from home_robot_hw.ros.camera import RosCamera
from home_robot_hw.ros.lidar import RosLidar
from home_robot_hw.ros.utils import matrix_from_pose_msg
from home_robot_hw.ros.visualizer import Visualizer

DEFAULT_COLOR_TOPIC = "/camera/color/image_raw"
DEFAULT_DEPTH_TOPIC = "/camera/aligned_depth_to_color/image_raw"
DEFAULT_LIDAR_TOPIC = "/scan"


class StretchRosInterface:
    """Interface object with ROS topics and services"""

    # Base of the robot
    base_link = "base_link"

    goal_time_tolerance = 1.0
    msg_delay_t = 0.1


    def __init__(
        self,
        node: Node,
        init_cameras: bool = True,
        color_topic: Optional[str] = None,
        depth_topic: Optional[str] = None,
        depth_buffer_size: Optional[int] = None,
        init_lidar: bool = True,
        lidar_topic: Optional[str] = None,
        verbose: bool = False,
    ):
        self.node = node
        # Verbosity for the ROS client
        self.verbose = verbose

        self.se3_base_filtered: Optional[sp.SE3] = None
        self.se3_base_odom: Optional[sp.SE3] = None
        self.se3_camera_pose: Optional[sp.SE3] = None
        self.at_goal: bool = False

        self.last_odom_update_timestamp = Time(seconds=0, nanoseconds=0)
        self.last_base_update_timestamp = Time(seconds=0, nanoseconds=0)
        self._goal_reset_t = Time(seconds=0, nanoseconds=0)

        # Create visualizers for pose information
        self.goal_visualizer = Visualizer("command_pose", rgba=[1.0, 0.0, 0.0, 0.5])
        self.curr_visualizer = Visualizer("current_pose", rgba=[0.0, 0.0, 1.0, 0.5])

        # Initialize ros communication
        self._create_pubs_subs()
        self._create_services()


        # Initialize cameras
        self._color_topic = DEFAULT_COLOR_TOPIC if color_topic is None else color_topic
        self._depth_topic = DEFAULT_DEPTH_TOPIC if depth_topic is None else depth_topic
        self._lidar_topic = DEFAULT_LIDAR_TOPIC if lidar_topic is None else lidar_topic
        self._depth_buffer_size = depth_buffer_size

        self.rgb_cam, self.dpt_cam = None, None
        if init_cameras:
            self._create_cameras()
            self._wait_for_cameras()
        # if init_lidar:
        #     self._lidar = RosLidar(self._lidar_topic)
        #     self._lidar.wait_for_scan()

    # Interfaces

    def recent_depth_image(self, seconds, print_delay_timers: bool = False):
        """Return true if we have up to date depth."""
        # Make sure we have a goal and our poses and depths are synced up - we need to have
        # received depth after we stopped moving
        if print_delay_timers:
            print(
                " - 1",
                (self.node.get_clock().now() - self._goal_reset_t).to_sec(),
                self.msg_delay_t,
            )
            print(
                " - 2", (self.dpt_cam.get_time() - self._goal_reset_t).to_sec(), seconds
            )
        if (
            self._goal_reset_t is not None
            and (self.node.get_clock().now() - self._goal_reset_t).to_sec() > self.msg_delay_t
        ):
            return (self.dpt_cam.get_time() - self._goal_reset_t).to_sec() > seconds
        else:
            return False


    # Helper functions


    def _create_services(self):
        """Create services to activate/deactive robot modes"""
         # 创建服务客户端
        self.goto_on_service: Client = self.node.create_client(Trigger, "goto_controller/enable")
        self.set_yaw_service: Client = self.node.create_client(SetBool, "goto_controller/set_yaw_tracking")

        # 等待服务可用
        print("Wait for mode service...")
        if not self.goto_on_service.wait_for_service(timeout_sec=5.0):
            self.node.get_logger().error('Service goto_controller/enable is not available.')
        if not self.set_yaw_service.wait_for_service(timeout_sec=5.0):
            self.node.get_logger().error('Service goto_controller/set_yaw_tracking is not available.')

    def _create_pubs_subs(self):
        """create ROS publishers and subscribers - only call once"""
        # Create the tf2 buffer first, used in camera init
        self.tf2_buffer = tf2_ros.Buffer()
        self.tf2_listener = tf2_ros.TransformListener(self.tf2_buffer)

        # Create command publishers
        self.goal_pub = self.node.create_publisher(Pose, "goto_controller/goal", 1)
        self.velocity_pub = self.node.create_publisher(Twist, "/cmd_vel", 1)

        # Create subscribers
        self._odom_sub = self.node.create_subscriber(
            Odometry,
            "odom",
            self._odom_callback,
            1,
        )
        self._base_state_sub = self.node.create_subscriber(
            PoseStamped,
            "state_estimator/pose_filtered",
            self._base_state_callback,
            1,
        )
        self._camera_pose_sub = self.node.create_subscriber(
            PoseStamped, "camera_pose", self._camera_pose_callback, 1
        )
        self._at_goal_sub = self.node.create_subscriber(
            Bool, "goto_controller/at_goal", self._at_goal_callback, 10
        )


    def _create_cameras(self):
        if self.rgb_cam is not None or self.dpt_cam is not None:
            raise RuntimeError("Already created cameras")
        print("Creating cameras...")
        self.rgb_cam = RosCamera(
            self._color_topic, rotations=0
        )
        self.dpt_cam = RosCamera(
            self._depth_topic,
            rotations=0,
            buffer_size=self._depth_buffer_size,
        )
        self.filter_depth = self._depth_buffer_size is not None

    def _wait_for_lidar(self):
        """wait until lidar has a message"""
        self._lidar.wait_for_scan()

    def _wait_for_cameras(self):
        if self.rgb_cam is None or self.dpt_cam is None:
            raise RuntimeError("cameras not initialized")
        print("Waiting for rgb camera images...")
        self.rgb_cam.wait_for_image()
        print("Waiting for depth camera images...")
        self.dpt_cam.wait_for_image()
        print("..done.")
        if self.verbose:
            print("rgb frame =", self.rgb_cam.get_frame())
            print("dpt frame =", self.dpt_cam.get_frame())
        # if self.rgb_cam.get_frame() != self.dpt_cam.get_frame():
        #     raise RuntimeError("issue with camera setup; depth and rgb not aligned")

    # Rostopic callbacks

    def _at_goal_callback(self, msg):
        """Is the velocity controller done moving; is it at its goal?"""
        self.at_goal = msg.data
        if not self.at_goal:
            self._goal_reset_t = None
        elif self._goal_reset_t is None:
            self._goal_reset_t = self.node.get_clock().now()


    def _odom_callback(self, msg: Odometry):
        """odometry callback"""
        self._last_odom_update_timestamp = msg.header.stamp
        self.se3_base_odom = sp.SE3(matrix_from_pose_msg(msg.pose.pose))
        # state estimator
        self.se3_base_filtered = self.se3_base_odom
        self._last_base_update_timestamp = self._last_odom_update_timestamp

    def _base_state_callback(self, msg: PoseStamped):
        """base state updates from SLAM system"""
        self._last_base_update_timestamp = msg.header.stamp
        self.se3_base_filtered = sp.SE3(matrix_from_pose_msg(msg.pose))
        self.curr_visualizer(self.se3_base_filtered.matrix())

    def _camera_pose_callback(self, msg: PoseStamped):
        """camera pose from CameraPosePublisher, which reads from tf"""
        self._last_camera_update_timestamp = msg.header.stamp
        self.se3_camera_pose = sp.SE3(matrix_from_pose_msg(msg.pose))


    def get_frame_pose(self, frame, base_frame=None, lookup_time=None, timeout_s=None):
        """look up a particular frame in base coords (or some other coordinate frame)."""
        if lookup_time is None:
            lookup_time = Time(seconds=0, nanoseconds=0)  # return most recent transform
        if timeout_s is None:
            timeout_ros = Duration(seconds=0.1)
        else:
            timeout_ros = Duration(seconds=timeout_s)
        if base_frame is None:
            base_frame = self.base_link
        try:
            stamped_transform = self.tf2_buffer.lookup_transform(
                base_frame, frame, lookup_time, timeout_ros
            )
            pose_mat = ros_numpy.numpify(stamped_transform.transform)
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ):
            print("!!! Lookup failed from", base_frame, "to", frame, "!!!")
            return None
        return pose_mat

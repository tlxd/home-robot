# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import Dict, Iterable, List, Optional

import numpy as np
import rospy
import torch

from home_robot.core.interfaces import Observations

from home_robot.utils.geometry import xyt2sophus

from .modules.head import StretchHeadClient
from .modules.nav import StretchNavigationClient
from .ros import StretchRosInterface


class StretchClient():
    """Defines a ROS-based interface to the real Stretch robot. Collect observations and command the robot."""

    def __init__(
        self,
        init_node: bool = True,
        camera_overrides: Optional[Dict] = None
    ):
        """Create an interface into ROS execution here. This one needs to connect to:
            - joint_states to read current position
            - tf for SLAM
            - FollowJointTrajectory for arm motions

        Based on this code:
        https://github.com/hello-robot/stretch_ros/blob/master/hello_helpers/src/hello_helpers/hello_misc.py
        """
        # Ros
        if init_node:
            rospy.init_node("stretch_user_client")

        if camera_overrides is None:
            camera_overrides = {}
        self._ros_client = StretchRosInterface(**camera_overrides)


        # Interface modules
        self.nav = StretchNavigationClient(self._ros_client)
        self.head = StretchHeadClient(self._ros_client)


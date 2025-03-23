# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from hashlib import blake2b
from typing import Any, Dict, Optional

import numpy as np

import home_robot
from home_robot.core.interfaces import Action, DiscreteNavigationAction, Observations,HybridAction
from home_robot.perception.detection.detic.detic_perception import DeticPerception
from home_robot.utils.geometry import xyt2sophus
from home_robot_hw.env.visualizer import Visualizer
from home_robot_hw.remote import StretchClient

REAL_WORLD_CATEGORIES = [
    "other",
    "tank", #"rifle", "tank"
    "other",
]


class StretchObjectNavEnv():
    """Create a detic-based object nav environment"""

    def __init__(
        self, config=None, goal_options = ["other", "tank", "other"], forward_step=0.25, rotate_step=30.0
    ):

        # TODO: pass this in or load from cfg
        self.goal_options = goal_options
        self.forward_step = forward_step  # in meters
        self.rotate_step = np.radians(rotate_step)

        # TODO Specify confidence threshold as a parameter
        self.segmentation = DeticPerception(
            vocabulary="custom",
            custom_vocabulary=",".join(self.goal_options),
            sem_gpu_id=0,
            confidence_threshold=0.45
        )
        self.visualizer = Visualizer(config)
        self.start = 0


        # Create a robot model, but we never need to visualize
        self.robot = StretchClient(init_node=False)
        self.reset()

    def reset(self):
        self.sample_goal()
        self._episode_start_pose = xyt2sophus(self.robot.nav.get_base_pose())
        self.visualizer.reset()


    def apply_action(
        self,
        next_act,
        action: Action,
        info: Optional[Dict[str, Any]] = None,
        prev_obs: Optional[Observations] = None,
    ):
        """Discrete action space. make predictions for where the robot should go, move by a fixed
        amount forward or rotationally."""
        if not isinstance(action, HybridAction):
            action = HybridAction(action)
        # Update the visualizer
        if self.visualizer is not None and info is not None:
            self.visualizer.visualize(**info)
        # Handle discrete actions first
        if action.is_discrete():
            action = action.get()
            continuous_action = np.zeros(3)
            if action == DiscreteNavigationAction.MOVE_FORWARD:
                print("[ENV] Move forward")
                continuous_action[0] = self.forward_step
            elif action == DiscreteNavigationAction.TURN_RIGHT:
                print("[ENV] TURN RIGHT")
                continuous_action[2] = -self.rotate_step
            elif action == DiscreteNavigationAction.TURN_LEFT:
                print("[ENV] Turn left")
                continuous_action[2] = self.rotate_step
            else:
                return True
        elif action.is_navigation():
            continuous_action = action.get()

        # Move, if we are not doing anything with the arm
        block = False
        if next_act != None :
            block = True
            # self.start += 1

        self.robot.nav.navigate_to(
                    continuous_action, relative=True, blocking=block
                )
        print("step action is ",continuous_action)
        return False


    def sample_goal(self):
        """set a random goal"""
        # idx = np.random.randint(len(self.goal_options) - 2) + 1
        idx = 1
        self.current_goal_id = idx
        self.current_goal_name = self.goal_options[idx]

    def get_observation(self) -> Observations:
        """Get Detic and rgb/xyz/theta from this"""
        rgb, depth, xyz = self.robot.head.get_images()
        current_pose = xyt2sophus(self.robot.nav.get_base_pose())

        # use sophus to get the relative translation
        relative_pose = self._episode_start_pose.inverse() * current_pose
        euler_angles = relative_pose.so3().log()
        theta = euler_angles[-1]
        # pos, vel, frc = self.get_joint_state()

        # GPS in robot coordinates
        gps = relative_pose.translation()[:2]

        # Create the observation
        obs = home_robot.core.interfaces.Observations(
            rgb=rgb.copy(),
            depth=depth.copy(),
            xyz=xyz.copy(),
            gps=gps,
            compass=np.array([theta]),
            # base_pose=sophus2obs(relative_pose),
            task_observations={
                "goal_id": self.current_goal_id,
                "goal_name": self.current_goal_name,
                "object_goal": self.current_goal_id,
                "recep_goal": self.current_goal_id,
            },
            camera_pose=relative_pose.matrix(),
        )
        # Run the segmentation model here
        obs = self.segmentation.predict(obs, depth_threshold=0.5)
        obs.semantic[obs.semantic != 1] = 4
        return obs

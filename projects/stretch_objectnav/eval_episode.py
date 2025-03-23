#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import rospy

import sys
sys.path.append('/home/lxd/HomeRobot/home-robot/src/home_robot') #TODO change the path

# from request_module import determine_target
from home_robot.agent.objectnav_agent.objectnav_agent import ObjectNavAgent
from home_robot.utils.config import get_config
sys.path.append('/home/lxd/HomeRobot/home-robot/src/home_robot_hw')
from home_robot_hw.env.stretch_object_nav_env import StretchObjectNavEnv
import time
def main( dry_run = False):
    config_path = "projects/stretch_objectnav/configs/agent/floorplanner_eval.yaml"
    config, config_str = get_config(config_path)
    config.defrost()
    config.NUM_ENVIRONMENTS = 1
    config.PRINT_IMAGES = 1
    config.EXP_NAME = "debug"
    config.freeze()

    rospy.init_node("eval_episode_stretch_objectnav")

    # instr = input("Please enter your instrution:")
    # goal = determine_target(instr) # Instr type: image path, description, category. Return: goal category
    # goal = instr
    # print(goal)
    pick_object = "other"
    start_recep = "tank"
    goal_recep = "other"

    agent = ObjectNavAgent(config=config)
    env = StretchObjectNavEnv(config=config, goal_options=[pick_object, start_recep, goal_recep])
    print(" agent reseting")
    agent.reset()
    # print("after agent reset")
    env.reset()
    # print('afeter env reset')
    t = 0
    while not rospy.is_shutdown():
        t += 1
        obs = env.get_observation()
        start = time.time()
        next_act, action, info = agent.act(obs)
        # print("Planning time: ", time.time() - start)
        print("Frequence: ", 1/(time.time() - start), " HZ")

        print("STEP =", t)
        stop = env.apply_action(next_act, action, info=info)
        if stop:
            break

if __name__ == "__main__":
    main()
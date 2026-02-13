#!/usr/bin/env python3

# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.

from typing import List, Optional

import numpy as np

from stretch.agent.operations import (
    NavigateToTagOperation,
    ScanTagsOperation,
    TagServoGraspOperation,
    TagServoPlaceOperation,
)
from stretch.agent.robot_agent import RobotAgent


class StackBlocksTask:
    """Stack blocks by detecting AprilTags and placing blocks on a base tag."""

    def __init__(
        self,
        agent: RobotAgent,
        base_tag_id: int,
        stack_tag_ids: Optional[List[int]] = None,
        tag_family: str = "apriltag_36h11",
        tag_size_m: float = 0.04,
        max_blocks: Optional[int] = None,
        use_update_scan: bool = False,
    ):
        self.agent = agent
        self.base_tag_id = base_tag_id
        self.stack_tag_ids = stack_tag_ids or []
        self.tag_family = tag_family
        self.tag_size_m = tag_size_m
        self.max_blocks = max_blocks
        self.use_update_scan = use_update_scan

    def _get_pose_world(self, tag_id: int):
        if not hasattr(self.agent, "tag_map"):
            return None
        obs = self.agent.tag_map.get(tag_id)
        if obs is None:
            return None
        return obs.pose_world

    def _derive_stack_list(self) -> List[int]:
        if self.stack_tag_ids:
            return list(self.stack_tag_ids)
        if not hasattr(self.agent, "tag_map"):
            return []
        tag_ids = [tid for tid in self.agent.tag_map.keys() if tid != self.base_tag_id]
        base_pose = self._get_pose_world(self.base_tag_id)
        if base_pose is None:
            return tag_ids
        base_xy = base_pose[:2, 3]
        tag_ids.sort(
            key=lambda tid: np.linalg.norm(self.agent.tag_map[tid].pose_world[:2, 3] - base_xy)
            if self.agent.tag_map[tid].pose_world is not None
            else float("inf")
        )
        return tag_ids

    def run(self) -> bool:
        scan = ScanTagsOperation("scan_tags", self.agent)
        if not scan(
            base_tag_id=self.base_tag_id,
            stack_tag_ids=self.stack_tag_ids,
            tag_family=self.tag_family,
            tag_size_m=self.tag_size_m,
            use_update_scan=self.use_update_scan,
        ):
            self.agent.robot_say("I could not find any tags.")
            return False

        if hasattr(self.agent, "tag_map") and len(self.agent.tag_map) > 0:
            print("Detected tags (id -> world xyz):")
            for tid, obs in sorted(self.agent.tag_map.items(), key=lambda kv: kv[0]):
                if obs.pose_world is None:
                    print(f"- {tid}: pose_world=None")
                else:
                    xyz = obs.pose_world[:3, 3]
                    print(f"- {tid}: [{xyz[0]:.3f}, {xyz[1]:.3f}, {xyz[2]:.3f}]")

        if self._get_pose_world(self.base_tag_id) is None:
            self.agent.robot_say("I could not find the base tag.")
            return False

        stack_list = self._derive_stack_list()
        if self.max_blocks is not None:
            stack_list = stack_list[: self.max_blocks]

        if len(stack_list) == 0:
            self.agent.robot_say("No blocks found to stack.")
            return False

        self.agent.stack_top_tag_id = self.base_tag_id

        for tag_id in stack_list:
            if tag_id == self.base_tag_id:
                continue

            nav_to_block = NavigateToTagOperation("navigate_to_block", self.agent)
            if not nav_to_block(
                tag_id=tag_id,
                rotation_offset=np.pi / 2,
                radius_m=self.agent.manipulation_radius,
                face_target=False,
            ):
                return False

            grasp = TagServoGraspOperation("tag_grasp", self.agent)
            refresh_tag_ids = [self.base_tag_id] + [tid for tid in stack_list if tid != self.base_tag_id]
            if not grasp(
                tag_id=tag_id,
                tag_family=self.tag_family,
                tag_size_m=self.tag_size_m,
                refresh_tag_ids=refresh_tag_ids,
                refresh_before_pre_detect=True,
                refresh_scan_tries=6,
            ):
                return False

            nav_to_stack = NavigateToTagOperation("navigate_to_stack", self.agent)
            if not nav_to_stack(
                tag_id=self.agent.stack_top_tag_id,
                rotation_offset=np.pi / 2,
                radius_m=self.agent.manipulation_radius,
                face_target=False,
            ):
                return False

            place = TagServoPlaceOperation("tag_place", self.agent)
            if not place(
                stack_top_tag_id=self.agent.stack_top_tag_id,
                tag_family=self.tag_family,
                tag_size_m=self.tag_size_m,
                refresh_tag_ids=refresh_tag_ids,
                refresh_before_place_detect=True,
                refresh_scan_tries=6,
            ):
                return False

            # Update top tag to the block we just placed
            self.agent.stack_top_tag_id = tag_id

        self.agent.robot.move_to_nav_posture()
        
        return True

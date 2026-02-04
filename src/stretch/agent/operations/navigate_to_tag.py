#!/usr/bin/env python3

# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.

import math
from typing import Optional

import numpy as np

from stretch.agent.base import ManagedOperation


class NavigateToTagOperation(ManagedOperation):
    """Navigate to a tag using its world pose stored in agent.tag_map."""

    def configure(
        self,
        tag_id: int,
        xy_margin: float = 0.35,
        z_margin: float = 0.15,
        rotation_offset: float = 0.0,
        be_precise: bool = False,
    ):
        self.tag_id = tag_id
        self.xy_margin = xy_margin
        self.z_margin = z_margin
        self.rotation_offset = rotation_offset
        self.be_precise = be_precise

    def can_start(self) -> bool:
        if not hasattr(self.agent, "tag_map"):
            self.error("No tag map available on agent.")
            return False
        if self.tag_id not in self.agent.tag_map:
            self.error(f"Tag {self.tag_id} not in tag map.")
            return False
        tag_obs = self.agent.tag_map[self.tag_id]
        if tag_obs.pose_world is None:
            self.error(f"Tag {self.tag_id} has no world pose.")
            return False
        return True

    def _build_bounds(self, pose_world: np.ndarray) -> np.ndarray:
        center = pose_world[:3, 3]
        bounds = np.zeros((3, 2), dtype=np.float32)
        bounds[0] = [center[0] - self.xy_margin, center[0] + self.xy_margin]
        bounds[1] = [center[1] - self.xy_margin, center[1] + self.xy_margin]
        bounds[2] = [center[2] - self.z_margin, center[2] + self.z_margin]
        return bounds

    def run(self) -> None:
        self.intro(f"Navigating to tag {self.tag_id}.")
        tag_obs = self.agent.tag_map[self.tag_id]
        bounds = self._build_bounds(tag_obs.pose_world)

        start = self.robot.get_base_pose()
        res = self.agent.plan_to_bounds(
            bounds,
            start,
            verbose=False,
            radius_m=0.7,
            rotation_offset=self.rotation_offset,
        )
        if not res.success:
            self.error("Failed to plan to tag bounds.")
            self._success = False
            return

        self.robot.move_to_nav_posture()
        self.robot.execute_trajectory(res, final_timeout=10.0)

        # Face the tag after navigation
        if tag_obs.pose_world is not None:
            xyt = self.robot.get_base_pose()
            tag_xy = tag_obs.pose_world[:2, 3]
            yaw = math.atan2(tag_xy[1] - xyt[1], tag_xy[0] - xyt[0])
            self.robot.move_base_to(np.array([xyt[0], xyt[1], yaw]), blocking=True, timeout=20.0)

        if self.be_precise:
            xyt = self.robot.get_base_pose()
            self.robot.move_base_to(xyt, blocking=True, timeout=20.0)

        self._success = True

    def was_successful(self) -> bool:
        return getattr(self, "_success", False)

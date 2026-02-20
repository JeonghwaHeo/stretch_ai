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
        xy_margin: float = 0.25,
        rotation_offset: float = 0.0,
        radius_m: float = 0.66,
        face_target: bool = True,
        prefer_closest_goal: bool = True,
        goal_sample_count: int = 500,
        max_plan_tries: int = 50,
        fallback_non_conservative: bool = True,
        fallback_goal_sample_count: Optional[int] = None,
        fallback_max_plan_tries: Optional[int] = None,
    ):
        self.tag_id = tag_id
        self.xy_margin = xy_margin
        self.rotation_offset = rotation_offset
        self.radius_m = radius_m
        self.face_target = face_target
        self.prefer_closest_goal = prefer_closest_goal
        self.goal_sample_count = goal_sample_count
        self.max_plan_tries = max_plan_tries
        self.fallback_non_conservative = bool(fallback_non_conservative)
        self.fallback_goal_sample_count = fallback_goal_sample_count
        self.fallback_max_plan_tries = fallback_max_plan_tries

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
        # Navigation uses a 2D (x, y, theta) state; the current mask builder ignores z.
        # Keep a degenerate z range to satisfy downstream bounds shape assertions.
        bounds[2] = [center[2], center[2]]
        return bounds

    def _plan_to_bounds_prefer_closest(
        self,
        bounds: np.ndarray,
        start: np.ndarray,
        conservative: bool = True,
        goal_sample_count: Optional[int] = None,
        max_plan_tries: Optional[int] = None,
    ):
        voxel_map = self.agent.get_voxel_map()
        if voxel_map is None:
            return None

        goal_sample_count = (
            self.goal_sample_count if goal_sample_count is None else int(goal_sample_count)
        )
        max_plan_tries = self.max_plan_tries if max_plan_tries is None else int(max_plan_tries)

        mask = voxel_map.mask_from_bounds(bounds)
        goals = []
        for goal in self.navigation_space.sample_near_mask(
            mask,
            radius_m=self.radius_m,
            conservative=conservative,
            rotation_offset=self.rotation_offset,
        ):
            goal_np = goal.cpu().numpy()
            goals.append(goal_np)
            if len(goals) >= goal_sample_count:
                break

        if len(goals) == 0:
            return None

        goals.sort(key=lambda g: float(np.linalg.norm(g[:2] - start[:2])))

        tries = 0
        for goal in goals:
            if tries >= max_plan_tries:
                break
            tries += 1
            if not self.navigation_space.is_valid(goal, verbose=False):
                continue
            res = self.agent.planner.plan(start, goal, verbose=False)
            if res is not None and res.success:
                return res

        return None

    def run(self) -> None:
        self.intro(f"Navigating to tag {self.tag_id}.")
        tag_obs = self.agent.tag_map[self.tag_id]
        bounds = self._build_bounds(tag_obs.pose_world)

        start = self.robot.get_base_pose()
        res = None
        if self.prefer_closest_goal:
            res = self._plan_to_bounds_prefer_closest(bounds, start, conservative=True)
        if res is None:
            res = self.agent.plan_to_bounds(
                bounds,
                start,
                verbose=False,
                radius_m=self.radius_m,
                rotation_offset=self.rotation_offset,
            )

        # Retry once with less conservative goal sampling if primary planning failed.
        if (res is None or not res.success) and self.fallback_non_conservative:
            self.warn("Primary plan failed (conservative=True). Retrying with conservative=False.")
            if self.prefer_closest_goal:
                res = self._plan_to_bounds_prefer_closest(
                    bounds,
                    start,
                    conservative=False,
                    goal_sample_count=self.fallback_goal_sample_count,
                    max_plan_tries=self.fallback_max_plan_tries,
                )
            if res is None:
                res = self.agent.plan_to_bounds(
                    bounds,
                    start,
                    verbose=False,
                    radius_m=self.radius_m,
                    rotation_offset=self.rotation_offset,
                )
            if res is not None and res.success:
                self.info("Fallback planning succeeded with conservative=False.")
            else:
                self.warn("Fallback planning failed.")

        if res is None or not res.success:
            self.error("Failed to plan to tag bounds.")
            self._success = False
            return

        self.robot.move_to_nav_posture()
        self.robot.execute_trajectory(res, final_timeout=10.0)

        # Optionally face the tag after navigation. For manipulation, leaving the planned
        # orientation (often with a +90deg offset) is important.
        if self.face_target and tag_obs.pose_world is not None:
            xyt = self.robot.get_base_pose()
            tag_xy = tag_obs.pose_world[:2, 3]
            yaw = math.atan2(tag_xy[1] - xyt[1], tag_xy[0] - xyt[0])
            self.robot.move_base_to(np.array([xyt[0], xyt[1], yaw]), blocking=True, timeout=20.0)

        self._success = True

    def was_successful(self) -> bool:
        return getattr(self, "_success", False)

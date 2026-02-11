#!/usr/bin/env python3

# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.

from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

from stretch.agent.operations import (
    GoToNavOperation,
    NavigateToTagOperation,
    RotateInPlaceOperation,
    ScanForTagsOperation,
    TagServoGraspOperation,
    TagServoPlaceOperation,
)
from stretch.agent.robot_agent import RobotAgent
from stretch.motion import constants
from stretch.motion.kinematics import HelloStretchIdx


class StackBlocksTask:
    """Stack blocks by detecting AprilTags and placing blocks on a base tag."""

    def __init__(
        self,
        agent: RobotAgent,
        base_tag_id: int,
        stack_tag_ids: Optional[List[int]] = None,
        tag_family: str = "apriltag_36h11",
        tag_size_m: float = 0.04,
        block_height_m: float = 0.05,
        max_blocks: Optional[int] = None,
        use_update_scan: bool = False,
    ):
        self.agent = agent
        self.base_tag_id = base_tag_id
        self.stack_tag_ids = stack_tag_ids or []
        self.tag_family = tag_family
        self.tag_size_m = tag_size_m
        self.block_height_m = block_height_m
        self.max_blocks = max_blocks
        self.use_update_scan = use_update_scan

    def _scan_tags(self) -> bool:
        go_nav = GoToNavOperation("go_to_nav", self.agent, retry_on_failure=True)
        if not go_nav():
            return False

        # Rotate in place and scan for tags at each step.
        steps = 8
        step_size = 2 * np.pi / steps
        x, y, theta = self.agent.robot.get_base_pose()
        full_sweep = True
        if full_sweep:
            steps += 1

        log_dir = Path("./logs/_scan_tags")
        log_dir.mkdir(parents=True, exist_ok=True)
        for image_path in log_dir.glob("*.png"):
            try:
                image_path.unlink()
            except OSError as exc:
                msg = f"Failed to remove old scan image {image_path}: {exc}"
                if hasattr(self.agent, "warn"):
                    self.agent.warn(msg)
                else:
                    print(f"[WARN] {msg}")

        any_tags = False
        target_tag_ids = None
        target_tag_obs: Dict[int, Optional[object]] = {}
        if len(self.stack_tag_ids) > 0:
            target_tag_ids = {self.base_tag_id, *self.stack_tag_ids}
            target_tag_obs = {tag_id: None for tag_id in target_tag_ids}
        # During scanning, tilt the head to -45 deg for better tag visibility.
        scan_head_pan = 0.0
        scan_head_tilt = float(np.deg2rad(-45.0))
        self.agent.robot.head_to(scan_head_pan, scan_head_tilt, blocking=True)
        for i in range(steps):
            self.agent.robot.move_base_to(
                [x, y, theta + (i * step_size)],
                relative=False,
                blocking=True,
                verbose=False,
            )
            if self.agent.robot.last_motion_failed():
                raise RuntimeError("Robot is stuck!")

            if not getattr(self.agent, "_realtime_updates", False):
                self.agent.update()

            obs = self.agent.robot.get_observation()
            if obs is not None and obs.rgb is not None:
                try:
                    save_ok = cv2.imwrite(str(log_dir / f"{i}.png"), obs.rgb)
                    if not save_ok:
                        msg = f"Failed to save scan image at step {i}."
                        if hasattr(self.agent, "warn"):
                            self.agent.warn(msg)
                        else:
                            print(f"[WARN] {msg}")
                except cv2.error as exc:
                    msg = f"OpenCV error while saving scan image at step {i}: {exc}"
                    if hasattr(self.agent, "warn"):
                        self.agent.warn(msg)
                    else:
                        print(f"[WARN] {msg}")
            else:
                msg = f"Missing head camera image at step {i}; skipping save."
                if hasattr(self.agent, "warn"):
                    self.agent.warn(msg)
                else:
                    print(f"[WARN] {msg}")

            max_scan_tries_per_step = 6
            if not hasattr(self.agent, "tag_map"):
                self.agent.tag_map = {}
            if not hasattr(self.agent, "tag_history"):
                self.agent.tag_history = []

            for scan_try in range(max_scan_tries_per_step):
                pre_hist_len = len(self.agent.tag_history)
                scan = ScanForTagsOperation("scan_for_tags", self.agent)
                scan(
                    tag_family=self.tag_family,
                    tag_size_m=self.tag_size_m,
                    use_update=self.use_update_scan,
                )
                try_obs = self.agent.tag_history[pre_hist_len:]
                try_count = len(try_obs)
                print(
                    f"scan step {i}: try {scan_try + 1}/{max_scan_tries_per_step}, detected={try_count}"
                )
                if try_count == 0:
                    continue

                if target_tag_ids is None:
                    any_tags = True
                    continue

                for obs_tag in try_obs:
                    if obs_tag.tag_id in target_tag_obs:
                        target_tag_obs[obs_tag.tag_id] = obs_tag
                        any_tags = True

            if target_tag_ids is None:
                continue

            found_count = sum(obs is not None for obs in target_tag_obs.values())
            print(
                f"scan step {i}: target tags found so far {found_count}/{len(target_tag_obs)}"
            )

        if target_tag_ids is not None:
            # Commit final target list observations so downstream uses consistent tag poses.
            if not hasattr(self.agent, "tag_map"):
                self.agent.tag_map = {}
            for tag_id, obs_tag in target_tag_obs.items():
                if obs_tag is not None:
                    self.agent.tag_map[tag_id] = obs_tag

        # Restore navigation head pose after scanning.
        nav_pan = float(constants.STRETCH_NAVIGATION_Q[HelloStretchIdx.HEAD_PAN])
        nav_tilt = float(constants.STRETCH_NAVIGATION_Q[HelloStretchIdx.HEAD_TILT])
        self.agent.robot.head_to(nav_pan, nav_tilt, blocking=True)

        return any_tags

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
        if not self._scan_tags():
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
            if not grasp(
                tag_id=tag_id,
                tag_family=self.tag_family,
                tag_size_m=self.tag_size_m,
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
                block_height_m=self.block_height_m,
            ):
                return False

            # Update top tag to the block we just placed
            self.agent.stack_top_tag_id = tag_id

        return True

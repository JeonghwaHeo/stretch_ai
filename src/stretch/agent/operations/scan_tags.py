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

from stretch.agent.base import ManagedOperation
from stretch.motion import constants
from stretch.motion.kinematics import HelloStretchIdx

from .switch_mode import GoToNavOperation
from .tag_detection import ScanForTagsOperation


class ScanTagsOperation(ManagedOperation):
    """Rotate in place and scan tags with the head camera."""

    def configure(
        self,
        base_tag_id: int,
        stack_tag_ids: Optional[List[int]] = None,
        tag_family: str = "apriltag_36h11",
        tag_size_m: float = 0.04,
        use_update_scan: bool = False,
        steps: int = 8,
        scan_tries_per_step: int = 6,
        scan_head_tilt_deg: float = -45.0,
        save_debug_images: bool = True,
    ):
        self.base_tag_id = base_tag_id
        self.stack_tag_ids = stack_tag_ids or []
        self.tag_family = tag_family
        self.tag_size_m = tag_size_m
        self.use_update_scan = use_update_scan
        self.steps = steps
        self.scan_tries_per_step = scan_tries_per_step
        self.scan_head_tilt_deg = scan_head_tilt_deg
        self.save_debug_images = save_debug_images

    def can_start(self) -> bool:
        return True

    def _prepare_log_dir(self) -> Optional[Path]:
        if not self.save_debug_images:
            return None
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
        return log_dir

    def _save_step_image(self, step_idx: int, log_dir: Optional[Path]) -> None:
        if log_dir is None:
            return
        obs = self.agent.robot.get_observation()
        if obs is not None and obs.rgb is not None:
            try:
                save_ok = cv2.imwrite(str(log_dir / f"scan_tags_{step_idx}.png"), obs.rgb)
                if not save_ok:
                    msg = f"Failed to save scan image at step {step_idx}."
                    if hasattr(self.agent, "warn"):
                        self.agent.warn(msg)
                    else:
                        print(f"[WARN] {msg}")
            except cv2.error as exc:
                msg = f"OpenCV error while saving scan image at step {step_idx}: {exc}"
                if hasattr(self.agent, "warn"):
                    self.agent.warn(msg)
                else:
                    print(f"[WARN] {msg}")
        else:
            msg = f"Missing head camera image at step {step_idx}; skipping save."
            if hasattr(self.agent, "warn"):
                self.agent.warn(msg)
            else:
                print(f"[WARN] {msg}")

    def _scan_step_targets(
        self,
        step_idx: int,
        target_tag_ids: Optional[set],
        target_tag_obs: Dict[int, Optional[object]],
    ) -> bool:
        any_tags = False
        if not hasattr(self.agent, "tag_map"):
            self.agent.tag_map = {}
        if not hasattr(self.agent, "tag_history"):
            self.agent.tag_history = []

        for scan_try in range(self.scan_tries_per_step):
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
                f"scan step {step_idx}: try {scan_try + 1}/{self.scan_tries_per_step}, detected={try_count}"
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
        return any_tags

    def run(self) -> None:
        go_nav = GoToNavOperation("go_to_nav", self.agent, retry_on_failure=True)
        if not go_nav():
            self._success = False
            return

        step_size = 2 * np.pi / self.steps
        x, y, theta = self.agent.robot.get_base_pose()
        full_sweep = True
        steps = self.steps + 1 if full_sweep else self.steps

        log_dir = self._prepare_log_dir()
        any_tags = False

        target_tag_ids = None
        target_tag_obs: Dict[int, Optional[object]] = {}
        if len(self.stack_tag_ids) > 0:
            target_tag_ids = {self.base_tag_id, *self.stack_tag_ids}
            target_tag_obs = {tag_id: None for tag_id in target_tag_ids}

        scan_head_pan = 0.0
        scan_head_tilt = float(np.deg2rad(self.scan_head_tilt_deg))
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

            self._save_step_image(i, log_dir)

            step_any_tags = self._scan_step_targets(i, target_tag_ids, target_tag_obs)
            any_tags = any_tags or step_any_tags

            if target_tag_ids is None:
                continue

            found_count = sum(obs is not None for obs in target_tag_obs.values())
            print(
                f"scan step {i}: target tags found so far {found_count}/{len(target_tag_obs)}"
            )

        if target_tag_ids is not None:
            if not hasattr(self.agent, "tag_map"):
                self.agent.tag_map = {}
            for tag_id, obs_tag in target_tag_obs.items():
                if obs_tag is not None:
                    self.agent.tag_map[tag_id] = obs_tag

        nav_pan = float(constants.STRETCH_NAVIGATION_Q[HelloStretchIdx.HEAD_PAN])
        nav_tilt = float(constants.STRETCH_NAVIGATION_Q[HelloStretchIdx.HEAD_TILT])
        self.agent.robot.head_to(nav_pan, nav_tilt, blocking=True)
        self._success = any_tags

    def was_successful(self) -> bool:
        return getattr(self, "_success", False)

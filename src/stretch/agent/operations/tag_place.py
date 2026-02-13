#!/usr/bin/env python3

# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.

from copy import deepcopy
from pathlib import Path
import time
from typing import Dict, List, Optional

import cv2
import numpy as np

import stretch.motion.constants as constants
from stretch.agent.base import ManagedOperation
from stretch.motion.kinematics import HelloStretchIdx

from .tag_detection import DetectAprilTagsOperation


class TagServoPlaceOperation(ManagedOperation):
    """Place a grasped block on top of a tag using the head camera."""

    def configure(
        self,
        stack_top_tag_id: int,
        tag_family: str = "apriltag_36h11",
        tag_size_m: float = 0.04,
        gripper_open_value: float = 0.8,
        place_offset: float = 0.97,
        offset_toward_robot: float = 0.46,
        refresh_tag_ids: Optional[List[int]] = None,
        refresh_before_place_detect: bool = True,
        refresh_scan_tries: int = 6,
        pre_place_head_servo_enable: bool = True,
        pre_place_head_servo_target_camera_x: float = 0.0,
        pre_place_head_servo_tol_camera_x: float = 0.01,
        pre_place_head_servo_max_steps: int = 30,
        pre_place_head_servo_max_misses: int = 8,
        pre_place_head_servo_gain_camera_x_to_base_x: float = 0.6,
        pre_place_head_servo_step_limit_base_x_m: float = 0.02,
    ):
        self.stack_top_tag_id = stack_top_tag_id
        self.tag_family = tag_family
        self.tag_size_m = tag_size_m
        self.gripper_open_value = float(gripper_open_value)
        self.place_offset = float(place_offset)
        self.offset_toward_robot = float(offset_toward_robot)
        self.refresh_tag_ids = refresh_tag_ids
        self.refresh_before_place_detect = bool(refresh_before_place_detect)
        self.refresh_scan_tries = int(refresh_scan_tries)
        self.pre_place_head_servo_enable = bool(pre_place_head_servo_enable)
        self.pre_place_head_servo_target_camera_x = float(pre_place_head_servo_target_camera_x)
        self.pre_place_head_servo_tol_camera_x = float(pre_place_head_servo_tol_camera_x)
        self.pre_place_head_servo_max_steps = int(pre_place_head_servo_max_steps)
        self.pre_place_head_servo_max_misses = int(pre_place_head_servo_max_misses)
        self.pre_place_head_servo_gain_camera_x_to_base_x = float(
            pre_place_head_servo_gain_camera_x_to_base_x
        )
        self.pre_place_head_servo_step_limit_base_x_m = float(pre_place_head_servo_step_limit_base_x_m)

    def can_start(self) -> bool:
        return True

    @staticmethod
    def _clamp(val: float, limit: float) -> float:
        return float(np.clip(val, -limit, limit))

    def _servo_align_base_x_to_camera_x(self) -> bool:
        if not self.robot.in_manipulation_mode():
            self.robot.switch_to_manipulation_mode()

        misses = 0
        for i in range(self.pre_place_head_servo_max_steps):
            tag_cam_pose = None
            detector = DetectAprilTagsOperation(
                "detect_apriltag_for_place_head_servo",
                agent=self.agent,
            )
            detector(
                tag_family=self.tag_family,
                tag_size_m=self.tag_size_m,
                camera="head",
                store_in_agent=False,
            )
            for obs in detector.get_observations():
                if obs.tag_id == self.stack_top_tag_id:
                    tag_cam_pose = obs.pose_camera
                    break
            if tag_cam_pose is None:
                misses += 1
                self.info(
                    f"Pre-place head servo step {i + 1}/{self.pre_place_head_servo_max_steps}: "
                    f"tag missing ({misses}/{self.pre_place_head_servo_max_misses})"
                )
                if misses >= self.pre_place_head_servo_max_misses:
                    self.warn("Pre-place head servo stopped: too many tag misses.")
                    return False
                continue
            misses = 0

            cam_x = float(tag_cam_pose[0, 3])
            err_x = cam_x - self.pre_place_head_servo_target_camera_x
            if abs(err_x) <= self.pre_place_head_servo_tol_camera_x:
                self.info(
                    f"Pre-place head servo converged at step {i + 1}: "
                    f"cam_x={cam_x:.4f}, err_x={err_x:.4f}"
                )
                return True

            step_base_x = self._clamp(
                -self.pre_place_head_servo_gain_camera_x_to_base_x * err_x,
                self.pre_place_head_servo_step_limit_base_x_m,
            )
            joint_state = self.robot.get_joint_positions().copy()
            joint_state[HelloStretchIdx.BASE_X] += step_base_x
            self.info(
                f"Pre-place head servo step {i + 1}/{self.pre_place_head_servo_max_steps}: "
                f"cam_x={cam_x:.4f}, err_x={err_x:.4f}, step_base_x={step_base_x:.4f}"
            )
            self.robot.arm_to(joint_state, head=constants.look_at_ee, blocking=True)

        self.warn("Pre-place head servo did not converge within max steps.")
        return False

    def _refresh_tag_map_with_head_scan(
        self, refresh_tag_ids: set[int], tries: int
    ) -> Dict[int, object]:
        if tries <= 0 or len(refresh_tag_ids) == 0:
            return {}

        log_dir = Path("./logs/_scan_tags")
        log_dir.mkdir(parents=True, exist_ok=True)

        if not hasattr(self.agent, "tag_map"):
            self.agent.tag_map = {}
        if not hasattr(self.agent, "tag_history"):
            self.agent.tag_history = []
        self._latest_stack_top_tag_cam_pose = None

        updated_obs_by_id: Dict[int, object] = {}
        xyz_samples_by_id: Dict[int, List[np.ndarray]] = {}
        cam_xyz_samples_top: List[np.ndarray] = []
        cam_pose_top_last: Optional[np.ndarray] = None
        for scan_try in range(tries):
            obs = self.robot.get_observation()
            if obs is not None and getattr(obs, "rgb", None) is not None:
                save_path = log_dir / f"tag_place_{scan_try}.png"
                save_ok = cv2.imwrite(str(save_path), obs.rgb)
                if not save_ok:
                    self.warn(f"Failed to save debug head image to {save_path}")
            else:
                self.warn(f"Missing head camera image for debug save at try {scan_try}")

            detector = DetectAprilTagsOperation(
                f"detect_apriltag_head_refresh_{scan_try}",
                agent=self.agent,
            )
            detector(
                tag_family=self.tag_family,
                tag_size_m=self.tag_size_m,
                camera="head",
                store_in_agent=False,
            )
            detections = detector.get_observations()
            matched_in_try = 0
            for obs in detections:
                if obs.tag_id not in refresh_tag_ids or obs.pose_world is None:
                    continue
                updated_obs_by_id[obs.tag_id] = obs
                xyz_samples_by_id.setdefault(obs.tag_id, []).append(obs.pose_world[:3, 3].copy())
                if obs.tag_id == self.stack_top_tag_id and obs.pose_camera is not None:
                    cam_xyz_samples_top.append(obs.pose_camera[:3, 3].copy())
                    cam_pose_top_last = obs.pose_camera
                matched_in_try += 1
            self.info(
                f"Head refresh try {scan_try + 1}/{tries}: detected={len(detections)} matched={matched_in_try}"
            )

        for tag_id, obs in updated_obs_by_id.items():
            obs_to_commit = obs
            xyz_samples = xyz_samples_by_id.get(tag_id, [])
            if len(xyz_samples) > 0:
                mean_xyz = np.mean(np.stack(xyz_samples, axis=0), axis=0)
                obs_to_commit = deepcopy(obs)
                obs_to_commit.pose_world = obs.pose_world.copy()
                obs_to_commit.pose_world[:3, 3] = mean_xyz

            self.agent.tag_map[tag_id] = obs_to_commit
            self.agent.tag_history.append(obs_to_commit)
            xyz = obs_to_commit.pose_world[:3, 3]
            self.info(
                f"Updated tag {tag_id}: world=[{xyz[0]:.3f}, {xyz[1]:.3f}, {xyz[2]:.3f}] "
                f"(avg from {len(xyz_samples)} detections)"
            )

        if cam_pose_top_last is not None and len(cam_xyz_samples_top) > 0:
            mean_cam_xyz = np.mean(np.stack(cam_xyz_samples_top, axis=0), axis=0)
            cam_pose_avg = cam_pose_top_last.copy()
            cam_pose_avg[:3, 3] = mean_cam_xyz
            self._latest_stack_top_tag_cam_pose = cam_pose_avg
            self.info(
                f"Updated stack-top camera pose from {len(cam_xyz_samples_top)} detections: "
                f"cam=[{mean_cam_xyz[0]:.3f}, {mean_cam_xyz[1]:.3f}, {mean_cam_xyz[2]:.3f}]"
            )

        self.info(
            f"Head refresh summary: updated {len(updated_obs_by_id)}/{len(refresh_tag_ids)} target tags"
        )
        return updated_obs_by_id

    def run(self) -> None:
        self.intro(f"Placing on tag {self.stack_top_tag_id}.")
        self._success = False

        self.robot.move_to_manip_posture()

        # Wait for motion to complete
        time.sleep(3.0)

        if self.refresh_before_place_detect:
            refresh_ids = set(self.refresh_tag_ids or [])
            if len(refresh_ids) == 0:
                refresh_ids = {self.stack_top_tag_id}
            self._refresh_tag_map_with_head_scan(refresh_ids, self.refresh_scan_tries)

        if self.pre_place_head_servo_enable:
            aligned = self._servo_align_base_x_to_camera_x()
            if not aligned:
                self.warn("Proceeding with place even though pre-place head servo did not converge.")

        if self.refresh_before_place_detect:
            refresh_ids = set(self.refresh_tag_ids or [])
            if len(refresh_ids) == 0:
                refresh_ids = {self.stack_top_tag_id}
            self._refresh_tag_map_with_head_scan(refresh_ids, self.refresh_scan_tries)

        tag_cam_pose = getattr(self, "_latest_stack_top_tag_cam_pose", None)
        if tag_cam_pose is None:
            self.error("Failed to get averaged stack top tag camera pose after refresh.")
            return

        cam_x, cam_y, cam_z = [float(v) for v in tag_cam_pose[:3, 3]]
        sqrt2 = np.sqrt(2.0)
        delta_lift = -(cam_y + cam_z) / sqrt2 + self.place_offset
        delta_arm = (cam_z - cam_y) / sqrt2 - self.offset_toward_robot
        self.info(
            f"Place camera pose cam=[{cam_x:.3f}, {cam_y:.3f}, {cam_z:.3f}] "
            f"-> delta_lift={delta_lift:.3f}, delta_arm={delta_arm:.3f}"
        )

        if not self.robot.in_manipulation_mode():
            self.robot.switch_to_manipulation_mode()
        q_target = self.robot.get_joint_positions().copy()
        q_target[HelloStretchIdx.LIFT] += delta_lift
        q_target[HelloStretchIdx.ARM] += delta_arm
        q_target[HelloStretchIdx.WRIST_ROLL] = 0.0
        q_target[HelloStretchIdx.WRIST_PITCH] = -np.deg2rad(45.0)
        q_target[HelloStretchIdx.WRIST_YAW] = 0.0
        self.info(
            f"Move target joints: lift={q_target[HelloStretchIdx.LIFT]:.3f}, "
            f"arm={q_target[HelloStretchIdx.ARM]:.3f}"
        )
        self.robot.arm_to(q_target, head=constants.look_at_ee, blocking=True)

        # Lower lift slightly before opening gripper.
        pre_open = self.robot.get_joint_positions().copy()
        pre_open[HelloStretchIdx.LIFT] -= 0.07
        self.robot.arm_to(pre_open, head=constants.look_at_ee, blocking=True)

        # Open gripper to place
        self.robot.gripper_to(self.gripper_open_value, blocking=True)

        # Retract arm to minimum
        retracted = self.robot.get_joint_positions().copy()
        retracted[HelloStretchIdx.ARM] = 0.0
        self.robot.arm_to(retracted, head=constants.look_at_ee, blocking=True)

        self._success = True

    def was_successful(self) -> bool:
        return getattr(self, "_success", False)

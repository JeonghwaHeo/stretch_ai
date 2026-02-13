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
from stretch.utils.geometry import point_global_to_base

from .tag_detection import DetectAprilTagsOperation


class TagServoGraspOperation(ManagedOperation):
    """Grasp a block by locating its tag with the end-effector camera."""

    def configure(
        self,
        tag_id: int,
        tag_family: str = "apriltag_36h11",
        tag_size_m: float = 0.04,
        grasp_height_offset: float = 0.01,
        lift_distance: float = 0.20,
        approach_offset: float = 0.08,
        pre_detect_offset_z: float = 0.30,
        pre_detect_offset_toward_robot: float = 0.53,
        pre_detect_pitch_deg: float = 45.0,
        use_tag_servo: bool = True,
        tag_target_cam_xyz: tuple = (0.0, 0.02, 0.22),
        tag_tol_cam_xyz: tuple = (0.005, 0.005, 0.005),
        tag_servo_max_steps: int = 80,
        tag_servo_max_misses: int = 10,
        tag_servo_gain_xyz: tuple = (0.5, 0.5, 0.5),
        tag_servo_step_limits: tuple = (0.03, 0.03, 0.02),
        gripper_open_value: float = 0.8,
        gripper_close_value: float = 0.1,
        refresh_tag_ids: Optional[List[int]] = None,
        refresh_before_pre_detect: bool = True,
        refresh_scan_tries: int = 6,
    ):
        self.tag_id = tag_id
        self.tag_family = tag_family
        self.tag_size_m = tag_size_m
        self.grasp_height_offset = grasp_height_offset
        self.lift_distance = lift_distance
        self.approach_offset = approach_offset
        self.pre_detect_offset_z = pre_detect_offset_z
        self.pre_detect_offset_toward_robot = pre_detect_offset_toward_robot
        self.pre_detect_pitch_deg = pre_detect_pitch_deg
        self.use_tag_servo = use_tag_servo
        self.tag_target_cam_xyz = np.array(tag_target_cam_xyz, dtype=np.float32)
        self.tag_tol_cam_xyz = np.array(tag_tol_cam_xyz, dtype=np.float32)
        self.tag_servo_max_steps = tag_servo_max_steps
        self.tag_servo_max_misses = tag_servo_max_misses
        self.tag_servo_gain_xyz = np.array(tag_servo_gain_xyz, dtype=np.float32)
        self.tag_servo_step_limits = np.array(tag_servo_step_limits, dtype=np.float32)
        self.gripper_open_value = float(gripper_open_value)
        self.gripper_close_value = float(gripper_close_value)
        self.refresh_tag_ids = refresh_tag_ids
        self.refresh_before_pre_detect = bool(refresh_before_pre_detect)
        self.refresh_scan_tries = int(refresh_scan_tries)

    def can_start(self) -> bool:
        return True

    def _detect_tag_pose(self) -> Optional[np.ndarray]:
        detector = DetectAprilTagsOperation("detect_apriltag_for_grasp", agent=self.agent)
        detector(
            tag_family=self.tag_family,
            tag_size_m=self.tag_size_m,
            camera="ee",
            store_in_agent=False,
        )
        for obs in detector.get_observations():
            if obs.tag_id == self.tag_id and obs.pose_world is not None:
                return obs.pose_world
        return None

    def _detect_tag_pose_camera(self) -> Optional[np.ndarray]:
        detector = DetectAprilTagsOperation("detect_apriltag_for_servo", agent=self.agent)
        detector(
            tag_family=self.tag_family,
            tag_size_m=self.tag_size_m,
            camera="ee",
            store_in_agent=False,
        )
        for obs in detector.get_observations():
            if obs.tag_id == self.tag_id:
                return obs.pose_camera
        return None

    @staticmethod
    def _clamp(val: float, limit: float) -> float:
        return float(np.clip(val, -limit, limit))

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

        updated_obs_by_id: Dict[int, object] = {}
        xyz_samples_by_id: Dict[int, List[np.ndarray]] = {}
        for scan_try in range(tries):
            obs = self.robot.get_observation()
            if obs is not None and getattr(obs, "rgb", None) is not None:
                save_path = log_dir / f"tag_grasp_{scan_try}.png"
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

        self.info(
            f"Head refresh summary: updated {len(updated_obs_by_id)}/{len(refresh_tag_ids)} target tags"
        )
        return updated_obs_by_id

    def run(self) -> None:
        self.intro(f"Grasping using tag {self.tag_id}.")
        self._success = False

        # If we have a map pose for the tag, rotate base so the tag is on -Y (graspable side).
        if hasattr(self.agent, "tag_map") and self.tag_id in self.agent.tag_map:
            tag_pose_world = self.agent.tag_map[self.tag_id].pose_world
            if tag_pose_world is not None:
                xyt = self.robot.get_base_pose()
                tag_xy = tag_pose_world[:2, 3]
                yaw = np.arctan2(tag_xy[1] - xyt[1], tag_xy[0] - xyt[0]) + (np.pi / 2.0)
                self.robot.move_base_to(
                    np.array([xyt[0], xyt[1], yaw]),
                    blocking=True,
                    timeout=20.0,
                )

        # Move to manipulation mode
        self.robot.move_to_manip_posture()

        # Wait for motion to complete
        time.sleep(3.0)

        # Refresh tag map with head camera before pre-detect positioning.
        if self.refresh_before_pre_detect:
            refresh_ids = set(self.refresh_tag_ids or [])
            if len(refresh_ids) == 0:
                refresh_ids = {self.tag_id}
            self._refresh_tag_map_with_head_scan(refresh_ids, self.refresh_scan_tries)

        # Use the latest map tag pose (from head camera) to pre-position the end-effector.
        tag_pose_world = None
        if hasattr(self.agent, "tag_map") and self.tag_id in self.agent.tag_map:
            tag_pose_world = self.agent.tag_map[self.tag_id].pose_world

        # Switch to manipulation mode
        self.robot.switch_to_manipulation_mode()

        # Open gripper
        # HomeRobotZmqClient.open_gripper() uses a preset and does not accept a target value.
        # For a custom "open" value, use gripper_to().
        self.robot.gripper_to(self.gripper_open_value, blocking=True)

        if tag_pose_world is not None:
            # Compute a pre-detection pose: 10cm above tag and 10cm toward the robot.
            target_xyz_world = tag_pose_world[:3, 3].copy()
            xyt = self.robot.get_base_pose()
            relative_xyz = point_global_to_base(target_xyz_world, xyt)

            rel_xy = relative_xyz[:2]
            rel_norm = np.linalg.norm(rel_xy)
            if rel_norm > 1e-6:
                toward_robot = -rel_xy / rel_norm
            else:
                toward_robot = np.array([1.0, 0.0])

            pre_xyz = relative_xyz.copy()
            pre_xyz[0] += self.pre_detect_offset_toward_robot * toward_robot[0]
            pre_xyz[1] += self.pre_detect_offset_toward_robot * toward_robot[1]
            pre_xyz[2] += self.pre_detect_offset_z

            # Keep the current end-effector orientation; only move to pre_xyz.
            q_now = self.robot.get_joint_positions()
            _, ee_rot = self.robot_model.manip_fk(q_now)

            target_joints, _, _, success, _ = self.robot_model.manip_ik_for_grasp_frame(
                pre_xyz, ee_rot, q0=q_now
            )
            if success:
                # Enforce wrist convention explicitly for pre-detect view.
                target_joints[HelloStretchIdx.WRIST_ROLL] = 0.0
                target_joints[HelloStretchIdx.WRIST_PITCH] = -np.deg2rad(
                    self.pre_detect_pitch_deg
                )
                target_joints[HelloStretchIdx.WRIST_YAW] = 0.0

                self.robot.arm_to(target_joints, head=constants.look_at_ee, blocking=True)

        if self.use_tag_servo:
            # Servo loop uses arm_to, which requires manipulation mode. Some base motions may
            # temporarily switch modes, so we re-assert manipulation mode as needed.
            if not self.robot.in_manipulation_mode():
                self.robot.switch_to_manipulation_mode()

            misses = 0
            for i in range(self.tag_servo_max_steps):
                tag_cam_pose = self._detect_tag_pose_camera()
                if tag_cam_pose is None:
                    misses += 1
                    if misses >= self.tag_servo_max_misses:
                        self.error("Failed to detect tag during servo loop.")
                        return
                    continue
                misses = 0

                tag_cam_xyz = tag_cam_pose[:3, 3]
                self.info(
                    "Tag servo step "
                    f"{i}: tag_cam_xyz=[{tag_cam_xyz[0]:.3f}, {tag_cam_xyz[1]:.3f}, {tag_cam_xyz[2]:.3f}]"
                )
                err = tag_cam_xyz - self.tag_target_cam_xyz
                if np.all(np.abs(err) <= self.tag_tol_cam_xyz):
                    break

                joint_state = self.robot.get_joint_positions()
                base_x = joint_state[HelloStretchIdx.BASE_X]
                lift = joint_state[HelloStretchIdx.LIFT]
                arm = joint_state[HelloStretchIdx.ARM]

                # OpenCV camera axes: x right, y down, z forward.
                dx, dy, dz = err.tolist()

                # Desired mapping:
                # - camera x error -> base_x
                # - camera y/z error -> lift and arm (coupled)
                step_base_x = self._clamp(
                    -self.tag_servo_gain_xyz[0] * dx, self.tag_servo_step_limits[0]
                )
                # Coupled control per user request:
                # - if dy > 0: decrease lift and arm
                # - if dz > 0: increase lift and arm
                # Use half-weight contributions from dy and dz into lift/arm.
                lift_delta_raw = (-0.5 * self.tag_servo_gain_xyz[1] * dy) + (
                    -0.5 * self.tag_servo_gain_xyz[2] * dz
                )
                arm_delta_raw = (-0.5 * self.tag_servo_gain_xyz[1] * dy) + (
                    0.5 * self.tag_servo_gain_xyz[2] * dz
                )
                step_lift = self._clamp(lift_delta_raw, self.tag_servo_step_limits[1])
                step_arm = self._clamp(arm_delta_raw, self.tag_servo_step_limits[2])

                base_x += step_base_x
                arm += step_arm
                lift += step_lift

                if not self.robot.in_manipulation_mode():
                    self.robot.switch_to_manipulation_mode()

                self.robot.arm_to(
                    [base_x, lift, arm, 0.0, -np.deg2rad(self.pre_detect_pitch_deg), 0.0],
                    head=constants.look_at_ee,
                    blocking=True,
                )

        tag_pose_world = self._detect_tag_pose()
        if tag_pose_world is None:
            self.error("Failed to detect target tag with end-effector camera.")
            return

        # NOTE: Disabling the additional approach/final IK correction logic.
        # We rely on the tag servo loop to bring the gripper into a good grasp pose.
        #
        # # Compute target grasp pose above the tag center
        # target_xyz_world = tag_pose_world[:3, 3].copy()
        # # Tag is on top of the block; grasp should be half block height below the tag.
        # target_xyz_world[2] += self.grasp_height_offset
        #
        # xyt = self.robot.get_base_pose()
        # relative_xyz = point_global_to_base(target_xyz_world, xyt)
        #
        # joint_state = self.robot.get_joint_positions()
        # ee_pos, ee_rot = self.robot_model.manip_fk(joint_state)
        #
        # # Approach offset along current end-effector z axis
        # ee_rot_m = R.from_quat(ee_rot).as_matrix()
        # approach_dir = ee_rot_m[:, 2]
        # approach_xyz = relative_xyz - self.approach_offset * approach_dir
        #
        # target_joints, _, _, success, _ = self.robot_model.manip_ik_for_grasp_frame(
        #     approach_xyz, ee_rot, q0=joint_state
        # )
        # if not success:
        #     self.error("IK failed for approach pose.")
        #     return
        #
        # # Move to approach
        # self.robot.arm_to(target_joints, head=constants.look_at_ee, blocking=True)
        #
        # # Move to final grasp position
        # target_joints_final, _, _, success, _ = self.robot_model.manip_ik_for_grasp_frame(
        #     relative_xyz, ee_rot, q0=target_joints
        # )
        # if not success:
        #     self.error("IK failed for grasp pose.")
        #     return
        #
        # self.robot.arm_to(target_joints_final, head=constants.look_at_ee, blocking=True)
        self.robot.gripper_to(self.gripper_close_value, blocking=True)

        # Lift
        joint_state = self.robot.get_joint_positions()
        lifted = joint_state.copy()
        lifted[HelloStretchIdx.LIFT] += self.lift_distance
        self.robot.arm_to(lifted, head=constants.look_at_ee, blocking=True)

        # Retract arm to minimum after lifting.
        retracted = self.robot.get_joint_positions().copy()
        retracted[HelloStretchIdx.ARM] = 0.0
        self.robot.arm_to(retracted, head=constants.look_at_ee, blocking=True)

        self._success = True

    def was_successful(self) -> bool:
        return getattr(self, "_success", False)

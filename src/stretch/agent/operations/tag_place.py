#!/usr/bin/env python3

# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.

from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation as R

import stretch.motion.constants as constants
from stretch.agent.base import ManagedOperation
from stretch.motion.kinematics import HelloStretchIdx
from stretch.utils.geometry import point_global_to_base

from .tag_detection import DetectAprilTagsOperation


class TagServoPlaceOperation(ManagedOperation):
    """Place a grasped block on top of a tag using the head camera."""

    def configure(
        self,
        stack_top_tag_id: int,
        tag_family: str = "apriltag_36h11",
        tag_size_m: float = 0.04,
        block_height_m: float = 0.05,
        place_height_margin: float = 0.01,
        approach_offset: float = 0.08,
    ):
        self.stack_top_tag_id = stack_top_tag_id
        self.tag_family = tag_family
        self.tag_size_m = tag_size_m
        self.block_height_m = block_height_m
        self.place_height_margin = place_height_margin
        self.approach_offset = approach_offset

    def can_start(self) -> bool:
        return True

    def _detect_tag_pose(self) -> Optional[np.ndarray]:
        detector = DetectAprilTagsOperation("detect_apriltag_for_place", agent=self.agent)
        detector(
            tag_family=self.tag_family,
            tag_size_m=self.tag_size_m,
            camera="head",
            store_in_agent=False,
        )
        for obs in detector.get_observations():
            if obs.tag_id == self.stack_top_tag_id and obs.pose_world is not None:
                return obs.pose_world
        return None

    def run(self) -> None:
        self.intro(f"Placing on tag {self.stack_top_tag_id}.")
        self._success = False

        tag_pose_world = self._detect_tag_pose()
        if tag_pose_world is None:
            self.error("Failed to detect stack top tag with head camera.")
            return

        # Compute target placement pose above the tag
        target_xyz_world = tag_pose_world[:3, 3].copy()
        target_xyz_world[2] += self.block_height_m + self.place_height_margin

        self.robot.move_to_manip_posture()
        self.robot.switch_to_manipulation_mode()

        xyt = self.robot.get_base_pose()
        relative_xyz = point_global_to_base(target_xyz_world, xyt)

        joint_state = self.robot.get_joint_positions()
        ee_pos, ee_rot = self.robot_model.manip_fk(joint_state)

        ee_rot_m = R.from_quat(ee_rot).as_matrix()
        approach_dir = ee_rot_m[:, 2]
        approach_xyz = relative_xyz - self.approach_offset * approach_dir

        target_joints, _, _, success, _ = self.robot_model.manip_ik_for_grasp_frame(
            approach_xyz, ee_rot, q0=joint_state
        )
        if not success:
            self.error("IK failed for place approach pose.")
            return

        self.robot.arm_to(target_joints, head=constants.look_at_ee, blocking=True)

        target_joints_final, _, _, success, _ = self.robot_model.manip_ik_for_grasp_frame(
            relative_xyz, ee_rot, q0=target_joints
        )
        if not success:
            self.error("IK failed for place pose.")
            return

        self.robot.arm_to(target_joints_final, head=constants.look_at_ee, blocking=True)
        self.robot.open_gripper(blocking=True)

        # Retract/lift slightly
        lifted = target_joints_final.copy()
        lifted[HelloStretchIdx.LIFT] += 0.05
        self.robot.arm_to(lifted, head=constants.look_at_ee, blocking=True)

        self._success = True

    def was_successful(self) -> bool:
        return getattr(self, "_success", False)

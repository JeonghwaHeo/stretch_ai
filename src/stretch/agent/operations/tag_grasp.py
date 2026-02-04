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


class TagServoGraspOperation(ManagedOperation):
    """Grasp a block by locating its tag with the end-effector camera."""

    def configure(
        self,
        tag_id: int,
        tag_family: str = "apriltag_36h11",
        tag_size_m: float = 0.04,
        grasp_height_offset: float = 0.02,
        lift_distance: float = 0.25,
        approach_offset: float = 0.08,
    ):
        self.tag_id = tag_id
        self.tag_family = tag_family
        self.tag_size_m = tag_size_m
        self.grasp_height_offset = grasp_height_offset
        self.lift_distance = lift_distance
        self.approach_offset = approach_offset

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

    def run(self) -> None:
        self.intro(f"Grasping using tag {self.tag_id}.")
        self._success = False

        tag_pose_world = self._detect_tag_pose()
        if tag_pose_world is None:
            self.error("Failed to detect target tag with end-effector camera.")
            return

        # Compute target grasp pose above the tag center
        target_xyz_world = tag_pose_world[:3, 3].copy()
        target_xyz_world[2] += self.grasp_height_offset

        # Move to manipulation mode
        self.robot.move_to_manip_posture()
        self.robot.switch_to_manipulation_mode()

        # Open gripper
        self.robot.open_gripper(blocking=True)

        xyt = self.robot.get_base_pose()
        relative_xyz = point_global_to_base(target_xyz_world, xyt)

        joint_state = self.robot.get_joint_positions()
        ee_pos, ee_rot = self.robot_model.manip_fk(joint_state)

        # Approach offset along current end-effector z axis
        ee_rot_m = R.from_quat(ee_rot).as_matrix()
        approach_dir = ee_rot_m[:, 2]
        approach_xyz = relative_xyz - self.approach_offset * approach_dir

        target_joints, _, _, success, _ = self.robot_model.manip_ik_for_grasp_frame(
            approach_xyz, ee_rot, q0=joint_state
        )
        if not success:
            self.error("IK failed for approach pose.")
            return

        # Move to approach
        self.robot.arm_to(target_joints, head=constants.look_at_ee, blocking=True)

        # Move to final grasp position
        target_joints_final, _, _, success, _ = self.robot_model.manip_ik_for_grasp_frame(
            relative_xyz, ee_rot, q0=target_joints
        )
        if not success:
            self.error("IK failed for grasp pose.")
            return

        self.robot.arm_to(target_joints_final, head=constants.look_at_ee, blocking=True)
        self.robot.close_gripper(blocking=True)

        # Lift
        lifted = target_joints_final.copy()
        lifted[HelloStretchIdx.LIFT] += self.lift_distance
        self.robot.arm_to(lifted, head=constants.look_at_ee, blocking=True)

        self._success = True

    def was_successful(self) -> bool:
        return getattr(self, "_success", False)

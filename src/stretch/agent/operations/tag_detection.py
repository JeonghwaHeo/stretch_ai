#!/usr/bin/env python3

# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

from dataclasses import dataclass
import os
from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from stretch.agent.base import ManagedOperation


@dataclass
class TagObservation:
    tag_id: int
    pose_camera: np.ndarray  # 4x4
    pose_world: Optional[np.ndarray]  # 4x4
    corners: np.ndarray  # 4x2
    center_px: Tuple[float, float]
    camera_name: str
    timestamp: float


def _tag_dictionary_for_family(tag_family: str):
    family_map = {
        "apriltag_36h11": "DICT_APRILTAG_36h11",
        "apriltag_25h9": "DICT_APRILTAG_25h9",
        "apriltag_16h5": "DICT_APRILTAG_16h5",
    }
    key = family_map.get(tag_family)
    if key is None:
        raise ValueError(f"Unsupported tag family: {tag_family}")
    if not hasattr(cv2.aruco, key):
        raise RuntimeError(f"OpenCV aruco does not support tag family {tag_family} ({key}).")
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, key))


def _estimate_tag_pose(
    corners: np.ndarray,
    tag_size_m: float,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    if hasattr(cv2.aruco, "estimatePoseSingleMarkers"):
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners.reshape(1, 4, 2), tag_size_m, camera_matrix, dist_coeffs
        )
        return rvecs[0][0], tvecs[0][0]

    # Fallback for OpenCV builds without estimatePoseSingleMarkers.
    half = tag_size_m / 2.0
    obj_pts = np.array(
        [
            [-half, half, 0.0],
            [half, half, 0.0],
            [half, -half, 0.0],
            [-half, -half, 0.0],
        ],
        dtype=np.float32,
    )
    img_pts = corners.reshape(4, 2).astype(np.float32)
    if hasattr(cv2, "SOLVEPNP_IPPE_SQUARE"):
        flag = cv2.SOLVEPNP_IPPE_SQUARE
    else:
        flag = cv2.SOLVEPNP_ITERATIVE
    ok, rvec, tvec = cv2.solvePnP(
        obj_pts,
        img_pts,
        camera_matrix,
        dist_coeffs,
        flags=flag,
    )
    if not ok:
        raise RuntimeError("solvePnP failed to estimate tag pose.")
    return rvec.reshape(-1), tvec.reshape(-1)


def _pose_from_rvec_tvec(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    rot, _ = cv2.Rodrigues(rvec)
    pose = np.eye(4)
    pose[:3, :3] = rot
    pose[:3, 3] = tvec
    return pose


class DetectAprilTagsOperation(ManagedOperation):
    """Detect AprilTags and update agent.tag_map."""

    def configure(
        self,
        tag_family: str = "apriltag_36h11",
        tag_size_m: float = 0.04,
        camera: str = "head",
        dist_coeffs: Optional[np.ndarray] = None,
        store_in_agent: bool = True,
    ):
        self.tag_family = tag_family
        self.tag_size_m = tag_size_m
        self.camera = camera
        self.dist_coeffs = dist_coeffs
        self.store_in_agent = store_in_agent

    def can_start(self) -> bool:
        return True

    def _get_observation(self):
        if self.camera == "ee":
            return self.robot.get_servo_observation()
        return self.robot.get_observation()

    def run(self) -> None:
        self.intro(f"Detecting AprilTags ({self.tag_family}) with {self.camera} camera.")
        self._observations: List[TagObservation] = []

        obs = self._get_observation()
        if self.camera == "ee":
            image = obs.ee_rgb
            camera_matrix = obs.ee_camera_K
            camera_pose = obs.ee_camera_pose
        else:
            image = obs.rgb
            camera_matrix = obs.camera_K
            camera_pose = obs.camera_pose

        if image is None or camera_matrix is None:
            self.error("Missing image or camera intrinsics for tag detection.")
            return

        dist_coeffs = (
            self.dist_coeffs
            if self.dist_coeffs is not None
            else np.zeros((5,), dtype=np.float32)
        )

        aruco_dict = _tag_dictionary_for_family(self.tag_family)
        aruco_params = cv2.aruco.DetectorParameters()
        aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = aruco_detector.detectMarkers(gray)

        if ids is None or len(ids) == 0:
            self.warn("No tags detected.")
            return

        for tag_corners, tag_id in zip(corners, ids.flatten()):
            rvec, tvec = _estimate_tag_pose(
                tag_corners, self.tag_size_m, camera_matrix, dist_coeffs
            )
            pose_camera = _pose_from_rvec_tvec(rvec, tvec)
            pose_world = None
            if camera_pose is not None:
                pose_world = camera_pose @ pose_camera
            center_px = tag_corners.reshape(4, 2).mean(axis=0)
            self._observations.append(
                TagObservation(
                    tag_id=int(tag_id),
                    pose_camera=pose_camera,
                    pose_world=pose_world,
                    corners=tag_corners.reshape(4, 2),
                    center_px=(float(center_px[0]), float(center_px[1])),
                    camera_name=self.camera,
                    timestamp=float(getattr(obs, "seq_id", -1)),
                )
            )

        # Print detections for debugging/visibility in apps like stack_blocks.
        self.info(f"Detected {len(self._observations)} tag(s):")
        for obs_tag in self._observations:
            cam_xyz = obs_tag.pose_camera[:3, 3]
            if obs_tag.pose_world is not None:
                world_xyz = obs_tag.pose_world[:3, 3]
                self.info(
                    f"- id={obs_tag.tag_id} world=[{world_xyz[0]:.3f}, {world_xyz[1]:.3f}, {world_xyz[2]:.3f}] "
                    f"cam=[{cam_xyz[0]:.3f}, {cam_xyz[1]:.3f}, {cam_xyz[2]:.3f}]"
                )
            else:
                self.info(
                    f"- id={obs_tag.tag_id} world=None cam=[{cam_xyz[0]:.3f}, {cam_xyz[1]:.3f}, {cam_xyz[2]:.3f}]"
                )

        if self.store_in_agent:
            if not hasattr(self.agent, "tag_map"):
                self.agent.tag_map = {}
            if not hasattr(self.agent, "tag_history"):
                self.agent.tag_history = []
            for obs_tag in self._observations:
                self.agent.tag_map[obs_tag.tag_id] = obs_tag
                self.agent.tag_history.append(obs_tag)

    def was_successful(self) -> bool:
        return hasattr(self, "_observations") and len(self._observations) > 0

    def get_observations(self) -> List[TagObservation]:
        return getattr(self, "_observations", [])


class ScanForTagsOperation(ManagedOperation):
    """Scan the scene and detect AprilTags using the head camera."""

    def configure(
        self,
        tag_family: str = "apriltag_36h11",
        tag_size_m: float = 0.04,
        use_update: bool = False,
    ):
        self.tag_family = tag_family
        self.tag_size_m = tag_size_m
        self.use_update = use_update

    def can_start(self) -> bool:
        return True

    def run(self) -> None:
        self.intro("Scanning for tags with head camera.")
        if self.use_update:
            self.update(move_head=True)
        detector = DetectAprilTagsOperation(
            name="detect_apriltags",
            agent=self.agent,
        )
        detector(
            tag_family=self.tag_family,
            tag_size_m=self.tag_size_m,
            camera="head",
            store_in_agent=True,
        )

    def was_successful(self) -> bool:
        return hasattr(self.agent, "tag_map") and len(self.agent.tag_map) > 0

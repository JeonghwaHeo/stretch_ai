#!/usr/bin/env python3

# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.

import click

from stretch.agent.robot_agent import RobotAgent
from stretch.agent.task.stack_blocks import StackBlocksTask
from stretch.agent.zmq_client import HomeRobotZmqClient
from stretch.core import get_parameters
from stretch.utils.logger import Logger

logger = Logger(__name__)


def _parse_tag_ids(tag_ids: str):
    if tag_ids is None or len(tag_ids.strip()) == 0:
        return []
    return [int(x.strip()) for x in tag_ids.split(",") if len(x.strip()) > 0]


@click.command()
@click.option("--robot_ip", default="", help="IP address of the robot")
@click.option("--reset", is_flag=True, help="Reset the robot to origin before starting")
@click.option(
    "--local",
    is_flag=True,
    help="Set if we are executing on the robot and not on a remote computer",
)
@click.option("--parameter_file", default="default_planner.yaml", help="Path to parameter file")
@click.option("--device_id", default=0, help="ID of the device to use for perception")
@click.option(
    "--realtime",
    "--real-time",
    "--enable-realtime-updates",
    "--enable_realtime_updates",
    is_flag=True,
    help="Enable real-time updates so that the robot will dynamically update its map",
)
@click.option("--tag_family", default="apriltag_36h11", help="AprilTag family to use")
@click.option("--tag_size_m", type=float, default=0.04, help="Tag size in meters")
@click.option("--block_height_m", type=float, default=0.05, help="Block height in meters")
@click.option("--base_tag_id", type=int, required=True, help="Tag ID to use as base block")
@click.option(
    "--stack_tag_ids",
    default="",
    help="Comma-separated tag IDs to stack, in order. If empty, uses detected tags.",
)
@click.option("--max_blocks", type=int, default=None, help="Maximum number of blocks to stack")
@click.option(
    "--use_update_scan",
    is_flag=True,
    help="Use agent.update() head sweep before detecting tags",
)
def main(
    robot_ip: str = "192.168.1.15",
    local: bool = False,
    parameter_file: str = "config/default_planner.yaml",
    device_id: int = 0,
    reset: bool = False,
    realtime: bool = False,
    tag_family: str = "apriltag_36h11",
    tag_size_m: float = 0.04,
    block_height_m: float = 0.05,
    base_tag_id: int = 0,
    stack_tag_ids: str = "",
    max_blocks: int = None,
    use_update_scan: bool = False,
):
    """Stack tagged blocks using AprilTag detection."""

    parameters = get_parameters(parameter_file)
    robot = HomeRobotZmqClient(
        robot_ip=robot_ip,
        use_remote_computer=(not local),
        parameters=parameters,
    )
    semantic_sensor = None
    agent = RobotAgent(robot, parameters, semantic_sensor, enable_realtime_updates=realtime)
    print("Starting robot agent: initializing...")
    agent.start(visualize_map_at_start=False)
    if reset:
        print("Reset: moving robot to origin")
        agent.move_closed_loop([0, 0, 0], max_time=60.0)

    tag_id_list = _parse_tag_ids(stack_tag_ids)

    task = StackBlocksTask(
        agent=agent,
        base_tag_id=base_tag_id,
        stack_tag_ids=tag_id_list,
        tag_family=tag_family,
        tag_size_m=tag_size_m,
        block_height_m=block_height_m,
        max_blocks=max_blocks,
        use_update_scan=use_update_scan,
    )

    ok = task.run()
    if not ok:
        logger.error("Stacking task failed.")

    robot.stop()


if __name__ == "__main__":
    main()

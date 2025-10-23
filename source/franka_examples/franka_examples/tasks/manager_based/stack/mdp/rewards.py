# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_ee_distance(
    env: ManagerBasedRLEnv,
    scale: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward the agent for reaching the object using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    # Target object position: (num_envs, 3)
    cube_pos_w = object.data.root_pos_w
    # End-effector position: (num_envs, 3)
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    # Tool-rightfinger position: (num_envs, 3)
    rf_w = ee_frame.data.target_pos_w[..., 1, :]
    # Tool-leftfinger position: (num_envs, 3)
    lf_w = ee_frame.data.target_pos_w[..., 2, :]
    # Distance of the end-effector to the object: (num_envs,)
    object_ee_distance = torch.norm(cube_pos_w - ee_w, dim=1)
    # Distance of the tool_rightfinger to the object: (num_envs,)
    object_rf_distance = torch.norm(cube_pos_w - rf_w, dim=1)
    # Distance of the tool_leftfinger to the object: (num_envs,)
    object_lf_distance = torch.norm(cube_pos_w - lf_w, dim=1)

    # Formula from video youtube: 1 - tanh(10*(d + d_lf + d_rf) / 3)
    distance_reward = 1 - torch.tanh(scale * (object_ee_distance + object_rf_distance + object_lf_distance) / 3.0)

    return torch.maximum(distance_reward, object_is_aligned(env)) * (1 - success_reward(env))


def object_is_lifted(
    env: ManagerBasedRLEnv,
    minimal_height: float = 0.04,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
) -> torch.Tensor:
    """Reward the agent for lifting the object above the minimal height."""
    object: RigidObject = env.scene[object_cfg.name]
    return torch.where(object.data.root_pos_w[:, 2] > minimal_height, 1.0, 0.0) * (1 - success_reward(env))


def object_is_aligned(
    env: ManagerBasedRLEnv,
    cube_1_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
    cube_2_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
    height_diff: float = 0.0466,
    scale: float = 10.0,
) -> torch.Tensor:
    """Reward the agent for aligning the object on top the target"""
    cube_1: RigidObject = env.scene[cube_1_cfg.name]
    cube_2: RigidObject = env.scene[cube_2_cfg.name]

    cube_1_pos = cube_1.data.root_pos_w
    cube_2_pos = cube_2.data.root_pos_w

    pos_diff = cube_1_pos - cube_2_pos

    pos_diff_with_offset = pos_diff.clone()
    pos_diff_with_offset[:, 2] += height_diff

    distance = torch.norm(pos_diff_with_offset, dim=1)

    is_lifted = object_is_lifted(env) 

    return (1 - torch.tanh(scale * distance)) * is_lifted * (1 - success_reward(env))

def success_reward(
    env: ManagerBasedRLEnv,
    cube_1_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
    cube_2_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    xy_threshold: float = 0.005,
    height_threshold: float = 0.005,
    height_diff: float = 0.0466,
    gripper_away_threshold: float = 0.1,
) -> torch.Tensor:
    """Reward the agent for successfully stacking the cubes."""
    cube_1: RigidObject = env.scene[cube_1_cfg.name]
    cube_2: RigidObject = env.scene[cube_2_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    # Compute cube position difference in x-y plane
    pos_diff_c12 = cube_1.data.root_pos_w - cube_2.data.root_pos_w
    xy_dist_c12 = torch.norm(pos_diff_c12[:, :2], dim=1)

    # Compute cube height difference
    h_dist_c12 = torch.abs(pos_diff_c12[:, 2] - height_diff)

    # Compute gripper away from cube
    ee_pos = ee_frame.data.target_pos_w[:, 0, :]
    cube_2_pos = cube_2.data.root_pos_w
    gripper_dist = torch.norm(ee_pos - cube_2_pos, dim=1)

    # Check cube positions
    stacked = torch.logical_and(xy_dist_c12 < xy_threshold, h_dist_c12 < height_threshold)
    stacked = torch.logical_and(gripper_dist > gripper_away_threshold, stacked)

    return stacked.to(torch.float32)

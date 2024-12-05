from dataclasses import dataclass
from typing import Any, Dict, Sequence, Tuple, Union, List

import numpy as np

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from functools import partial

from brax import math
import brax.base as base
from brax.base import System
from brax import envs as brax_envs
from brax.envs.base import PipelineEnv, State
from brax.io import html, mjcf, model

import mujoco
from mujoco import mjx

from dial_mpc.envs.base_env import BaseEnv, BaseEnvConfig
from dial_mpc.utils.function_utils import global_to_body_velocity, get_foot_step
from dial_mpc.utils.io_utils import get_model_path


@dataclass
class PandaEnvConfig(BaseEnvConfig):
    # Reward scaling for different components
    reward_scales: Dict[str, float] = None

    def __post_init__(self):
        # Default reward scales if none are provided
        if self.reward_scales is None:
            self.reward_scales = {
                "gripper_box": 4.0,
                "box_target": 8.0,
                "no_floor_collision": 0.25,
                "robot_target_qpos": 0.3,
            }


class PandaRobotEnv(BaseEnv):
    def __init__(self, config: PandaEnvConfig):
        super().__init__(config)
        self._reward_scales = config.reward_scales

        # Initialize constants and indices
        self._arm_joint_indices = list(range(7))
        self._finger_joint_indices = [7, 8]
        self._gripper_site = self.sys.mj_model.site("gripper").id
        self._box_body = self.sys.mj_model.body("box").id
        self._target_id = self.sys.mj_model.site("mocap_target_site").id
        self._init_box_pos = self.sys.mj_model.keyframe("home").qpos[9:12]
        self._lowers = self.sys.mj_model.actuator_ctrlrange[:, 0]
        self._uppers = self.sys.mj_model.actuator_ctrlrange[:, 1]

    def make_system(self, config: PandaEnvConfig) -> System:
        model_path = get_model_path("franka_emika_panda", "mjx_single_cube.xml")
        print(f"Model path: {model_path}") 
        sys = mjcf.load(model_path)
        
        sys = sys.tree_replace({"opt.timestep": config.timestep})
        return sys


    def reset(self, rng: jax.Array) -> State:
        rng, rng_box, rng_target = jax.random.split(rng, 3)

        # Randomize positions
        box_pos = jax.random.uniform(
            rng_box,
            (3,),
            minval=jnp.array([0.1, 0.2, 0.3]),  #  min values
            maxval=jnp.array([0.5, 0.6, 0.7]),  #  max values
        )

        target_pos = jax.random.uniform(
            rng_target,
            (3,),
            minval=jnp.array([-1.0, -1.0, 0.0]),  
            maxval=jnp.array([1.0, 1.0, 1.0]),    
        )


        init_q = jnp.array(self.sys.mj_model.keyframe("home").qpos).at[9:12].set(box_pos)

        pipeline_state = self.pipeline_init(init_q, jnp.zeros(self.sys.nv))

        info = {"rng": rng, "target_pos": target_pos}
        obs = self._get_obs(pipeline_state, info)
        return State(pipeline_state, obs, jnp.zeros(()), jnp.zeros(()), {}, info)

    def step(self, state: State, action: jax.Array) -> State:
        # Scale and clip actions
        delta = action * self._config.action_scale
        ctrl = state.pipeline_state.ctrl + delta
        ctrl = jnp.clip(ctrl, self._lowers, self._uppers)

        # Step the physics
        pipeline_state = self.pipeline_step(state.pipeline_state, ctrl)

        # Compute rewards
        target_pos = state.info["target_pos"]
        box_pos = pipeline_state.xpos[self._box_body]
        gripper_pos = pipeline_state.site_xpos[self._gripper_site]

        rewards = {
            "box_target": 1 - jnp.tanh(5 * jnp.linalg.norm(target_pos - box_pos)),
            "gripper_box": 1 - jnp.tanh(5 * jnp.linalg.norm(box_pos - gripper_pos)),
            "robot_target_qpos": 1 - jnp.tanh(
                jnp.linalg.norm(state.pipeline_state.qpos[jnp.array(self._arm_joint_indices)])

            ),
            "no_floor_collision": jnp.all(box_pos[2] > 0.0).astype(float),
        }
        reward = sum(self._reward_scales[k] * v for k, v in rewards.items())

        # Check termination conditions
        done = jnp.any(box_pos[2] < 0.0)

        # Update observation and metrics
        obs = self._get_obs(pipeline_state, state.info)
        metrics = {**rewards}
        return State(pipeline_state, obs, reward=reward, done=done, metrics=metrics, info=state.info)

    def _get_obs(self, pipeline_state: State, info: dict) -> jax.Array:
        gripper_pos = pipeline_state.site_xpos[self._gripper_site]
        box_pos = pipeline_state.xpos[self._box_body]
        target_pos = info["target_pos"]

        return jnp.concatenate(
            [
                pipeline_state.qpos,
                pipeline_state.qvel,
                gripper_pos,
                box_pos - gripper_pos,
                target_pos - box_pos,
            ]
        )



brax_envs.register_environment("PandaRobot", PandaRobotEnv)

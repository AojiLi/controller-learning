"""Challenge and Gymnasium environments."""

from controller_learning.envs.car_racing import CarRacingEnv
from controller_learning.envs.configuration import race_core_config_from_project
from controller_learning.envs.observation import (
    OBSERVATION_KEYS,
    VehicleStateView,
    action_space,
    batched_action_space,
    batched_observation_space,
    encode_batched_observation,
    observation_space,
    observation_to_host,
    unbatch_observation,
)
from controller_learning.envs.race_core import (
    RaceCoreConfig,
    RaceState,
    RaceStep,
    RaceTermination,
    TrackProjection,
    body_to_world,
    masked_reset_race_state,
    project_to_track,
    reset_race_state,
    step_race_core,
    world_to_body,
    wrap_angle,
)
from controller_learning.envs.registration import ENV_ID, register_environments
from controller_learning.envs.vector_racing import VecCarRacingEnv

register_environments()

__all__ = [
    "ENV_ID",
    "OBSERVATION_KEYS",
    "CarRacingEnv",
    "RaceCoreConfig",
    "RaceState",
    "RaceStep",
    "RaceTermination",
    "TrackProjection",
    "VecCarRacingEnv",
    "VehicleStateView",
    "action_space",
    "batched_action_space",
    "batched_observation_space",
    "body_to_world",
    "encode_batched_observation",
    "masked_reset_race_state",
    "observation_space",
    "observation_to_host",
    "project_to_track",
    "race_core_config_from_project",
    "register_environments",
    "reset_race_state",
    "step_race_core",
    "unbatch_observation",
    "world_to_body",
    "wrap_angle",
]

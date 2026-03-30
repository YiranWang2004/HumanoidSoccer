import gymnasium as gym

from . import agents, flat_env_cfg
from . import soccer_flat_env_cfg

##
# Register Gym environments.
##

## Motion tracking environments
gym.register(
    id="Tracking-Flat-PiPlus-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.PiPlusFlatEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PiPlusFlatPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-PiPlus-RNN-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.PiPlusFlatEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PiPlusFlatRecurrentPPORunnerCfg",
    },
)


## Soccer environments
### Stage 1 — Terrain
gym.register(
    id="Tracking-Terrain-PiPlus-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": soccer_flat_env_cfg.PiPlusTerrainMotionEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PiPlusFlatPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Terrain-PiPlus-RNN-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": soccer_flat_env_cfg.PiPlusTerrainMotionEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PiPlusFlatRecurrentPPORunnerCfg",
    },
)

### Stage 1 — Flat
gym.register(
    id="Tracking-Flat-PiPlus-Motion-RNN-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": soccer_flat_env_cfg.PiPlusFlatMotionEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PiPlusFlatRecurrentPPORunnerCfg",
    },
)


### Stage 2 — Kick
gym.register(
    id="Tracking-Flat-PiPlus-SoccerDestination-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": soccer_flat_env_cfg.PiPlusFlatKickEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PiPlusFlatPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-PiPlus-SoccerDestination-RNN-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": soccer_flat_env_cfg.PiPlusFlatKickEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PiPlusFlatRecurrentPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-PiPlus-SoccerMoving-RNN-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": soccer_flat_env_cfg.PiPlusFlatKickMovingEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PiPlusFlatRecurrentPPORunnerCfg",
    },
)


## Advanced Soccer environments

gym.register(
    id="Tracking-Flat-PiPlus-SoccerBlind-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": soccer_flat_env_cfg.PiPlusFlatSoccerBlindEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PiPlusFlatPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-PiPlus-SoccerBlind-RNN-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": soccer_flat_env_cfg.PiPlusFlatSoccerBlindEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PiPlusFlatRecurrentPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-PiPlus-SuperSoccer-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": soccer_flat_env_cfg.PiPlusFlatSuperSoccerEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PiPlusFlatPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-PiPlus-Soccer-Distillation-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": soccer_flat_env_cfg.PiPlusFlatSoccerStudentEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PiPlusFlatStudentTeacherPPORunnerCfg",
    },
)

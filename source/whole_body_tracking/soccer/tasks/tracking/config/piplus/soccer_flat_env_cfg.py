import math

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.markers import VisualizationMarkersCfg

from soccer.assets import ASSET_DIR
from soccer.tasks.tracking import mdp
from soccer.tasks.tracking.tracking_env_cfg import MySceneCfg
from soccer.tasks.tracking.config.piplus.flat_env_cfg import PiPlusFlatEnvCfg

import isaaclab.terrains as terrain_gen
from isaaclab.terrains import TerrainImporterCfg, TerrainGeneratorCfg


SOCCER_BALL_RADIUS = 0.11
SOCCER_ASSET_PATH = f"{ASSET_DIR}/soccer/soccer.usda"


def _apply_soccer_obs(cfg):
    cfg.observations.policy.target_point_pos = ObsTerm(
        func=mdp.constant_target_point_pos,
        params={"command_name": "motion"},
    )
    cfg.observations.critic.target_point_pos = ObsTerm(
        func=mdp.constant_target_point_pos,
        params={"command_name": "motion"},
    )
    cfg.observations.policy.target_destination_pos_local = ObsTerm(
        func=mdp.target_destination_pos_local,
        params={"command_name": "motion"},
    )
    cfg.observations.critic.target_destination_pos_local = ObsTerm(
        func=mdp.target_destination_pos_local,
        params={"command_name": "motion"},
    )


def _apply_soccer_scene(cfg):
    cfg.scene.soccer_ball = cfg.scene.soccer_ball.replace(prim_path="{ENV_REGEX_NS}/SoccerBall")
    cfg.scene.soccer_ball.init_state.pos = (0.0, 0.0, SOCCER_BALL_RADIUS)

    cfg.commands.motion.target_point_marker_cfg = VisualizationMarkersCfg(
        prim_path="/World/Visuals/TargetPoint",
        markers={
            "target_sphere": sim_utils.SphereCfg(
                radius=0.11,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
            ),
        },
    )
    cfg.commands.motion.target_destination_marker_cfg = VisualizationMarkersCfg(
        prim_path="/World/Visuals/PostKickTarget",
        markers={
            "destination_sphere": sim_utils.SphereCfg(
                radius=0.11,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
            ),
        },
    )


## Scene configuration

@configclass
class PiPlusFlatSoccerSceneCfg(MySceneCfg):
    def __post_init__(self):
        super().__post_init__()
        self.terrain.physics_material = self.terrain.physics_material.replace(restitution=0.8)

    soccer_ball = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/SoccerBall",
        spawn=sim_utils.UsdFileCfg(
            usd_path=SOCCER_ASSET_PATH,
            activate_contact_sensors=True,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.7, 0.0, SOCCER_BALL_RADIUS),
        ),
    )
    soccer_ball_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/SoccerBall",
        history_length=3,
        track_air_time=False,
        force_threshold=0.0,
        debug_vis=False,
    )


## Terrain environment

@configclass
class PiPlusTerrainEnvCfg(PiPlusFlatEnvCfg):

    def __post_init__(self):
        super().__post_init__()
        self.commands.motion.class_type = mdp.commands_multi_motion_soccer.MotionCommand
        self.terminations.anchor_pos_z = DoneTerm(
            func=mdp.bad_anchor_pos_z_only,
            params={"command_name": "motion", "threshold": 0.15},
        )
        self.terminations.anchor_ori = DoneTerm(
            func=mdp.bad_anchor_ori,
            params={"asset_cfg": SceneEntityCfg("robot"), "command_name": "motion", "threshold": 0.8},
        )
        self.terminations.ee_body_pos = DoneTerm(
            func=mdp.bad_motion_body_pos_z_only,
            params={
                "command_name": "motion",
                "threshold": 0.15,
                "body_names": [
                    "l_ankle_roll_link",
                    "r_ankle_roll_link",
                    "l_wrist_link",
                    "r_wrist_link",
                ],
            },
        )

        GRAVEL_TERRAINS_CFG = TerrainGeneratorCfg(
            curriculum=False,
            size=(8.0, 8.0),
            border_width=20.0,
            num_rows=10,
            num_cols=20,
            horizontal_scale=0.1,
            vertical_scale=0.005,
            slope_threshold=0.75,
            use_cache=False,
            sub_terrains={
                "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
                    proportion=1.0, noise_range=(-0.02, 0.02), noise_step=0.02, border_width=0.0
                )
            },
        )
        self.scene.terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="generator",
            terrain_generator=GRAVEL_TERRAINS_CFG,
        )


@configclass
class PiPlusTerrainMotionEnvCfg(PiPlusTerrainEnvCfg):
    scene: PiPlusFlatSoccerSceneCfg = PiPlusFlatSoccerSceneCfg(num_envs=4096, env_spacing=2.5)

    def __post_init__(self):
        super().__post_init__()
        _apply_soccer_obs(self)
        _apply_soccer_scene(self)


## Flat soccer environments

@configclass
class PiPlusFlatMotionEnvCfg(PiPlusFlatEnvCfg):
    scene: PiPlusFlatSoccerSceneCfg = PiPlusFlatSoccerSceneCfg(num_envs=4096, env_spacing=2.5)

    def __post_init__(self):
        super().__post_init__()
        self.commands.motion.class_type = mdp.commands_multi_motion_soccer.MotionCommand
        _apply_soccer_obs(self)
        _apply_soccer_scene(self)


@configclass
class PiPlusFlatProximityEnvCfg(PiPlusFlatMotionEnvCfg):

    def __post_init__(self):
        super().__post_init__()

        self.foot_cfg = SceneEntityCfg(
            "robot",
            body_names=[
                "l_ankle_roll_link",
                "r_ankle_roll_link",
            ],
        )

        # PiPlus only has waist_yaw (no roll/pitch)
        self.waist_cfg = SceneEntityCfg(
            "robot",
            joint_names=["waist_yaw_joint"],
        )

        self.commands.motion.curve_offset_range = {
            "radius": (-0.25, 0.25),
            "arc_angle": math.pi / 9,
            "height": SOCCER_BALL_RADIUS,
        }

        self.rewards.foot_distance = RewTerm(
            func=mdp.foot_distance,
            weight=0.2,
            params={
                "threshold": 0.24,
                "std": 0.5,
                "foot_cfg": self.foot_cfg,
            },
        )

        self.rewards.target_point_proximity = RewTerm(
            func=mdp.target_point_proximity,
            weight=1.0,
            params={
                "std": 4.0,
                "command_name": "motion",
            },
        )

        self.rewards.motion_global_anchor_pos = RewTerm(
            func=mdp.motion_global_anchor_position_error_exp,
            weight=0.0,
            params={"command_name": "motion", "std": 0.3},
        )

        self.rewards.motion_global_anchor_ori = RewTerm(
            func=mdp.motion_global_anchor_orientation_error_exp,
            weight=1.0,
            params={"command_name": "motion", "std": 0.4},
        )

        self.rewards.waist_action_rate_l2 = RewTerm(
            func=mdp.waist_action_rate_l2_clip,
            weight=-2.5e-1,
            params={
                "waist_cfg": self.waist_cfg,
            },
        )

        self.rewards.pelvis_orientation = RewTerm(
            func=mdp.pelvis_orientation,
            weight=-1.0,
            params={"command_name": "motion"},
        )

        self.rewards.motion_body_pos = RewTerm(
            func=mdp.motion_relative_body_position_error_exp,
            weight=1.0,
            params={
                "command_name": "motion",
                "std": 0.3,
                "body_names": [
                    "base_link",
                    "l_hip_roll_link",
                    "l_calf_link",
                    "r_hip_roll_link",
                    "r_calf_link",
                    "torso_link",
                    "l_shoulder_roll_link",
                    "l_elbow_link",
                    "l_wrist_link",
                    "r_shoulder_roll_link",
                    "r_elbow_link",
                    "r_wrist_link",
                ],
            },
        )

        self.motion_body_ori = RewTerm(
            func=mdp.motion_relative_body_orientation_error_exp,
            weight=1.0,
            params={
                "command_name": "motion",
                "std": 0.4,
                "body_names": [
                    "base_link",
                    "l_hip_roll_link",
                    "l_calf_link",
                    "r_hip_roll_link",
                    "r_calf_link",
                    "torso_link",
                    "l_shoulder_roll_link",
                    "l_elbow_link",
                    "l_wrist_link",
                    "r_shoulder_roll_link",
                    "r_elbow_link",
                    "r_wrist_link",
                ],
            },
        )

        self.rewards.motion_foot_pos = RewTerm(
            func=mdp.motion_relative_foot_position_error_exp,
            weight=1.0,
            params={
                "command_name": "motion",
                "std": 0.3,
                "foot_body_names": [
                    "l_ankle_roll_link",
                    "r_ankle_roll_link",
                ],
            },
        )


@configclass
class PiPlusFlatKickEnvCfg(PiPlusFlatProximityEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.rewards.target_point_contact = RewTerm(
            func=mdp.target_point_contact,
            weight=50.0,
            params={
                "command_name": "motion",
                "ball_sensor_name": "soccer_ball_contact",
                "horizontal_force_threshold": 10,
                "foot_cfg": self.foot_cfg,
            },
        )

        self.rewards.sideways_kick = RewTerm(
            func=mdp.sideways_kick,
            weight=50.0,
            params={
                "command_name": "motion",
                "ball_sensor_name": "soccer_ball_contact",
                "horizontal_force_threshold": 10,
                "foot_cfg": self.foot_cfg,
            },
        )

        self.rewards.ball_velocity_direction_alignment = RewTerm(
            func=mdp.ball_velocity_direction_alignment,
            weight=30.0,
            params={
                "command_name": "motion",
                "std": 0.8,
                "velocity_threshold": 0.5,
                "ball_sensor_name": "soccer_ball_contact",
                "horizontal_force_threshold": 10,
                "foot_cfg": self.foot_cfg,
            },
        )

        self.rewards.ball_speed_reward = RewTerm(
            func=mdp.ball_speed_reward,
            weight=10.0,
            params={
                "command_name": "motion",
                "std": 1.2,
                "velocity_threshold": 0.5,
                "ball_sensor_name": "soccer_ball_contact",
                "horizontal_force_threshold": 10,
                "foot_cfg": self.foot_cfg,
            },
        )

        self.rewards.ball_z_speed_penalty_reward = RewTerm(
            func=mdp.ball_z_speed_penalty_reward,
            weight=-0.0,
            params={
                "command_name": "motion",
                "std": 3,
                "velocity_threshold": 0.5,
            },
        )


@configclass
class PiPlusFlatKickMovingEnvCfg(PiPlusFlatKickEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.commands.motion.enable_soccer_ball_init_vel = True
        self.commands.motion.soccer_ball_init_lin_vel_range = {
            "x": (-0.3, 0.3),
            "y": (-0.3, 0.3),
            "z": (0.0, 0.0),
        }


@configclass
class PiPlusFlatSoccerBlindEnvCfg(PiPlusFlatKickEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.commands.motion.blind_distance_min_range = (0.2, 0.8)
        self.commands.motion.blind_distance_max_range = (1.8, 2.5)

        self.observations.policy.target_point_pos = ObsTerm(
            func=mdp.blind_zone_target_point_pos,
            params={"command_name": "motion"},
        )
        self.observations.critic.target_point_pos = ObsTerm(
            func=mdp.blind_zone_target_point_pos,
            params={"command_name": "motion"},
        )


@configclass
class PiPlusFlatSuperSoccerEnvCfg(PiPlusFlatKickEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.observations.policy.motion_anchor_pos_b = ObsTerm(
            func=mdp.motion_anchor_pos_b, params={"command_name": "motion"}
        )
        self.observations.policy.motion_anchor_ori_b = ObsTerm(
            func=mdp.motion_anchor_ori_b, params={"command_name": "motion"}
        )
        self.observations.policy.body_pos = ObsTerm(
            func=mdp.robot_body_pos_b, params={"command_name": "motion"}
        )
        self.observations.policy.body_ori = ObsTerm(
            func=mdp.robot_body_ori_b, params={"command_name": "motion"}
        )
        self.observations.policy.base_lin_vel = ObsTerm(func=mdp.base_lin_vel)

        self.observations.critic.projected_gravity = ObsTerm(func=mdp.projected_gravity)
        self.observations.critic.motion_ref_ang_vel = ObsTerm(
            func=mdp.motion_anchor_ang_vel, params={"command_name": "motion"}
        )


@configclass
class PiPlusFlatSoccerStudentEnvCfg(PiPlusFlatKickEnvCfg):

    def __post_init__(self):
        super().__post_init__()
        student_obs = self.observations.policy.copy()
        student_obs.target_point_pos = ObsTerm(
            func=mdp.target_point_pos_first_frame,
            params={"command_name": "motion"},
        )
        self.observations.StudentPolicyCfg = student_obs

        student_obs.target_destination_pos_local = ObsTerm(
            func=mdp.target_destination_pos_local_first_frame,
            params={"command_name": "motion"},
        )

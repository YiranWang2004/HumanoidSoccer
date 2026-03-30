from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

import soccer.tasks.tracking.mdp as mdp
from soccer.robots.piplus import PiPlus_S_12L8A0G2H1W_ACTION_SCALE, PiPlus_S_12L8A0G2H1W_CFG
from soccer.tasks.tracking.tracking_env_cfg import TrackingEnvCfg
from soccer.tasks.tracking.mdp import commands_multi_motion as motion_cmds


@configclass
class PiPlusFlatEnvCfg(TrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = PiPlus_S_12L8A0G2H1W_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = PiPlus_S_12L8A0G2H1W_ACTION_SCALE
        self.commands.motion.anchor_body_name = "torso_link"
        self.commands.motion.class_type = motion_cmds.MotionCommand
        self.commands.motion.body_names = [
            "base_link",
            "l_hip_roll_link",
            "l_calf_link",
            "l_ankle_roll_link",
            "r_hip_roll_link",
            "r_calf_link",
            "r_ankle_roll_link",
            "torso_link",
            "l_shoulder_roll_link",
            "l_elbow_link",
            "l_wrist_link",
            "r_shoulder_roll_link",
            "r_elbow_link",
            "r_wrist_link",
        ]

        # Override undesired_contacts with PiPlus link names (l_/r_ prefix, no wrist_yaw).
        # Allow contacts on ankle_roll links (feet) and wrist_links (hands).
        self.rewards.undesired_contacts = RewTerm(
            func=mdp.undesired_contacts,
            weight=-0.1,
            params={
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces",
                    body_names=[
                        r"^(?!l_ankle_roll_link$)(?!r_ankle_roll_link$)"
                        r"(?!l_wrist_link$)(?!r_wrist_link$).+$"
                    ],
                ),
                "threshold": 1.0,
            },
        )

        # PiPlus standing height is 0.351 m — use a smaller fall threshold.
        self.terminations.anchor_pos_z = DoneTerm(
            func=mdp.bad_anchor_pos_z_only,
            params={"command_name": "motion", "threshold": 0.15},
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

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from soccer.assets import ASSET_DIR

"""
joint name order (from isaacsim):
[
0: l_hip_pitch_joint
1: r_hip_pitch_joint
2: waist_yaw_joint
3: l_hip_roll_joint
4: r_hip_roll_joint
5: head_yaw_joint
6: l_shoulder_pitch_joint
7: r_shoulder_pitch_joint
8: l_thigh_joint
9: r_thigh_joint
10: head_pitch_joint
11: l_shoulder_roll_joint
12: r_shoulder_roll_joint
13: l_calf_joint
14: r_calf_joint
15: l_upper_arm_joint
16: r_upper_arm_joint
17: l_ankle_pitch_joint
18: r_ankle_pitch_joint
19: l_elbow_joint
20: r_elbow_joint
21: l_ankle_roll_joint
22: r_ankle_roll_joint
]

Policy controls 21 joints (head excluded):
Indices 0-4, 6-9, 11-22 (skipping 5=head_yaw, 10=head_pitch)
"""

# Armature values for different motor types
ARMATURE_4438 = 0.008419
ARMATURE_5047 = 0.044277
ARMATURE_3536 = 0.004383

NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10Hz
DAMPING_RATIO = 2.0

STIFFNESS_4438 = ARMATURE_4438 * NATURAL_FREQ**2
STIFFNESS_5047 = ARMATURE_5047 * NATURAL_FREQ**2
STIFFNESS_3536 = ARMATURE_3536 * NATURAL_FREQ**2

DAMPING_4438 = 2.0 * DAMPING_RATIO * ARMATURE_4438 * NATURAL_FREQ
DAMPING_5047 = 2.0 * DAMPING_RATIO * ARMATURE_5047 * NATURAL_FREQ
DAMPING_3536 = 2.0 * DAMPING_RATIO * ARMATURE_3536 * NATURAL_FREQ


PiPlus_S_12L8A0G2H1W_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        replace_cylinders_with_capsules=True,
        asset_path=f"{ASSET_DIR}/hightorque_description/urdf/piplus.urdf",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.351),
        joint_pos={
            ".*_hip_pitch_joint": -0.25,
            ".*_calf_joint": 0.65,
            ".*_ankle_pitch_joint": -0.4,
            ".*_elbow_joint": 0.0,
            "l_shoulder_roll_joint": 0.0,
            "l_shoulder_pitch_joint": 0.0,
            "r_shoulder_roll_joint": 0.0,
            "r_shoulder_pitch_joint": 0.0,
            "head_yaw_joint": 0.0,
            "head_pitch_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_thigh_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_calf_joint",
            ],
            effort_limit_sim={
                ".*_thigh_joint": 20.0,
                ".*_hip_roll_joint": 20.0,
                ".*_hip_pitch_joint": 20.0,
                ".*_calf_joint": 20.0,
            },
            velocity_limit_sim={
                ".*_thigh_joint": 100.0,
                ".*_hip_roll_joint": 100.0,
                ".*_hip_pitch_joint": 100.0,
                ".*_calf_joint": 100.0,
            },
            stiffness={
                ".*_hip_pitch_joint": STIFFNESS_5047,
                ".*_hip_roll_joint": STIFFNESS_5047,
                ".*_thigh_joint": STIFFNESS_5047,
                ".*_calf_joint": STIFFNESS_5047,
            },
            damping={
                ".*_hip_pitch_joint": DAMPING_5047,
                ".*_hip_roll_joint": DAMPING_5047,
                ".*_thigh_joint": DAMPING_5047,
                ".*_calf_joint": DAMPING_5047,
            },
            armature={
                ".*_hip_pitch_joint": ARMATURE_5047,
                ".*_hip_roll_joint": ARMATURE_5047,
                ".*_thigh_joint": ARMATURE_5047,
                ".*_calf_joint": ARMATURE_5047,
            },
        ),
        "feet": ImplicitActuatorCfg(
            effort_limit_sim=20.0,
            velocity_limit_sim=100.0,
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            stiffness=STIFFNESS_5047,
            damping=DAMPING_5047,
            armature=ARMATURE_5047,
        ),
        "waist_yaw": ImplicitActuatorCfg(
            effort_limit_sim=20.0,
            velocity_limit_sim=100.0,
            joint_names_expr=["waist_yaw_joint"],
            stiffness=STIFFNESS_5047,
            damping=DAMPING_5047,
            armature=ARMATURE_5047,
        ),
        "head": ImplicitActuatorCfg(
            effort_limit_sim=3.0,
            velocity_limit_sim=100.0,
            joint_names_expr=["head_yaw_joint", "head_pitch_joint"],
            stiffness=STIFFNESS_3536,
            damping=DAMPING_3536,
            armature=ARMATURE_3536,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_upper_arm_joint",
                ".*_elbow_joint",
            ],
            effort_limit_sim=20.0,
            velocity_limit_sim=100.0,
            stiffness=STIFFNESS_4438,
            damping=DAMPING_4438,
            armature=ARMATURE_4438,
        ),
    },
)

PiPlus_S_12L8A0G2H1W_ACTION_SCALE = {}
for a in PiPlus_S_12L8A0G2H1W_CFG.actuators.values():
    e = a.effort_limit_sim
    s = a.stiffness
    names = a.joint_names_expr
    if not isinstance(e, dict):
        e = {n: e for n in names}
    if not isinstance(s, dict):
        s = {n: s for n in names}
    for n in names:
        if n in e and n in s and s[n]:
            PiPlus_S_12L8A0G2H1W_ACTION_SCALE[n] = 0.25 * e[n] / s[n]


# Symmetric augmentation joint mapping for PiPlus S 12L8A0G2H1W
# Maps left-right joint pairs for data augmentation
PiPlus_S_12L8A0G2H1W_symmetric_augmentation_joint_mapping = [
    1,   # 0: l_hip_pitch_joint -> r_hip_pitch_joint
    0,   # 1: r_hip_pitch_joint -> l_hip_pitch_joint
    2,   # 2: waist_yaw_joint (stays same)
    4,   # 3: l_hip_roll_joint -> r_hip_roll_joint
    3,   # 4: r_hip_roll_joint -> l_hip_roll_joint
    5,   # 5: head_yaw_joint (stays same)
    7,   # 6: l_shoulder_pitch_joint -> r_shoulder_pitch_joint
    6,   # 7: r_shoulder_pitch_joint -> l_shoulder_pitch_joint
    9,   # 8: l_thigh_joint -> r_thigh_joint
    8,   # 9: r_thigh_joint -> l_thigh_joint
    10,  # 10: head_pitch_joint (stays same)
    12,  # 11: l_shoulder_roll_joint -> r_shoulder_roll_joint
    11,  # 12: r_shoulder_roll_joint -> l_shoulder_roll_joint
    14,  # 13: l_calf_joint -> r_calf_joint
    13,  # 14: r_calf_joint -> l_calf_joint
    16,  # 15: l_upper_arm_joint -> r_upper_arm_joint
    15,  # 16: r_upper_arm_joint -> l_upper_arm_joint
    18,  # 17: l_ankle_pitch_joint -> r_ankle_pitch_joint
    17,  # 18: r_ankle_pitch_joint -> l_ankle_pitch_joint
    20,  # 19: l_elbow_joint -> r_elbow_joint
    19,  # 20: r_elbow_joint -> l_elbow_joint
    22,  # 21: l_ankle_roll_joint -> r_ankle_roll_joint
    21,  # 22: r_ankle_roll_joint -> l_ankle_roll_joint
]

# Joint reverse buffer for symmetric augmentation
# 1 means keep sign, -1 means flip sign when mirroring
PiPlus_S_12L8A0G2H1W_symmetric_augmentation_joint_reverse_buf = [
    1,   # 0: l_hip_pitch_joint (keep)
    1,   # 1: r_hip_pitch_joint (keep)
    -1,  # 2: waist_yaw_joint (flip)
    -1,  # 3: l_hip_roll_joint (flip)
    -1,  # 4: r_hip_roll_joint (flip)
    -1,  # 5: head_yaw_joint (flip)
    1,   # 6: l_shoulder_pitch_joint (keep)
    1,   # 7: r_shoulder_pitch_joint (keep)
    -1,  # 8: l_thigh_joint (flip)
    -1,  # 9: r_thigh_joint (flip)
    1,   # 10: head_pitch_joint (keep)
    -1,  # 11: l_shoulder_roll_joint (flip)
    -1,  # 12: r_shoulder_roll_joint (flip)
    1,   # 13: l_calf_joint (keep)
    1,   # 14: r_calf_joint (keep)
    -1,  # 15: l_upper_arm_joint (flip)
    -1,  # 16: r_upper_arm_joint (flip)
    1,   # 17: l_ankle_pitch_joint (keep)
    1,   # 18: r_ankle_pitch_joint (keep)
    1,   # 19: l_elbow_joint (keep)
    1,   # 20: r_elbow_joint (keep)
    -1,  # 21: l_ankle_roll_joint (flip)
    -1,  # 22: r_ankle_roll_joint (flip)
]

PiPlus_S_12L8A0G2H1W_LINKS = [  # Order not guaranteed.
    "base_link",
    "waist_yaw_link",
    "head_yaw_link",
    "head_pitch_link",
    "l_hip_pitch_link",
    "l_hip_roll_link",
    "l_thigh_link",
    "l_calf_link",
    "l_ankle_pitch_link",
    "l_ankle_roll_link",
    "l_shoulder_pitch_link",
    "l_shoulder_roll_link",
    "l_upper_arm_link",
    "l_elbow_link",
    "r_hip_pitch_link",
    "r_hip_roll_link",
    "r_thigh_link",
    "r_calf_link",
    "r_ankle_pitch_link",
    "r_ankle_roll_link",
    "r_shoulder_pitch_link",
    "r_shoulder_roll_link",
    "r_upper_arm_link",
    "r_elbow_link",
]


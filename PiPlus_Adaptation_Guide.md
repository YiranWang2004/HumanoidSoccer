# PiPlus 机器人适配分析文档

> 本文档系统分析将 HumanoidSoccer 项目中的 Unitree G1 机器人替换为 PiPlus 机器人所需的全部适配工作。

---

## 一、机器人结构对比

### 1.1 关节结构总览（G1 vs PiPlus）

| 部位 | G1 关节名 | PiPlus 关节名 | 状态 |
|------|----------|--------------|------|
| 右髋 pitch | `right_hip_pitch_joint` | `r_hip_pitch_joint` | 前缀不同 |
| 右髋 roll | `right_hip_roll_joint` | `r_hip_roll_joint` | 前缀不同 |
| 右髋 yaw | `right_hip_yaw_joint` | `r_thigh_joint` | **命名完全不同** |
| 右膝 | `right_knee_joint` | `r_calf_joint` | **命名不同** |
| 右踝 pitch | `right_ankle_pitch_joint` | `r_ankle_pitch_joint` | 前缀不同 |
| 右踝 roll | `right_ankle_roll_joint` | `r_ankle_roll_joint` | 前缀不同 |
| 左髋 pitch | `left_hip_pitch_joint` | `l_hip_pitch_joint` | 前缀不同 |
| 左髋 roll | `left_hip_roll_joint` | `l_hip_roll_joint` | 前缀不同 |
| 左髋 yaw | `left_hip_yaw_joint` | `l_thigh_joint` | **命名完全不同** |
| 左膝 | `left_knee_joint` | `l_calf_joint` | **命名不同** |
| 左踝 pitch | `left_ankle_pitch_joint` | `l_ankle_pitch_joint` | 前缀不同 |
| 左踝 roll | `left_ankle_roll_joint` | `l_ankle_roll_joint` | 前缀不同 |
| 腰 yaw | `waist_yaw_joint` | `waist_yaw_joint` | **相同** |
| 右肩 pitch | `right_shoulder_pitch_joint` | `r_shoulder_pitch_joint` | 前缀不同 |
| 右肩 roll | `right_shoulder_roll_joint` | `r_shoulder_roll_joint` | 前缀不同 |
| 右肩 yaw | `right_shoulder_yaw_joint` | `r_upper_arm_joint` | **命名完全不同** |
| 右肘 | `right_elbow_joint` | `r_elbow_joint` | 前缀不同 |
| 右腕 roll | `right_wrist_roll_joint` | *(无)* | **PiPlus 缺失** |
| 右腕 pitch | `right_wrist_pitch_joint` | *(无)* | **PiPlus 缺失** |
| 右腕 yaw | `right_wrist_yaw_joint` | *(无)* | **PiPlus 缺失** |
| 左肩 pitch | `left_shoulder_pitch_joint` | `l_shoulder_pitch_joint` | 前缀不同 |
| 左肩 roll | `left_shoulder_roll_joint` | `l_shoulder_roll_link` | 前缀不同 |
| 左肩 yaw | `left_shoulder_yaw_joint` | `l_upper_arm_joint` | **命名完全不同** |
| 左肘 | `left_elbow_joint` | `l_elbow_joint` | 前缀不同 |
| 左腕 roll | `left_wrist_roll_joint` | *(无)* | **PiPlus 缺失** |
| 左腕 pitch | `left_wrist_pitch_joint` | *(无)* | **PiPlus 缺失** |
| 左腕 yaw | `left_wrist_yaw_joint` | *(无)* | **PiPlus 缺失** |

**G1: 29 DOF，PiPlus: 23 DOF**（少 6 个腕关节）


### 1.2 关键命名规则差异

| 维度 | G1 | PiPlus |
|------|----|--------|
| 左右前缀 | `left_` / `right_` | `l_` / `r_` |
| Hip yaw 关节 | `*_hip_yaw_joint` | `*_thigh_joint` |
| 膝关节 | `*_knee_joint` | `*_calf_joint` |
| 肩 yaw 关节 | `*_shoulder_yaw_joint` | `*_upper_arm_joint` |
| 腕关节 | 3 DOF per arm | **无** |

### 1.3 关键 Body（Link）名称对比

| 部位 | G1 | PiPlus |
|------|----|--------|
| 根/骨盆 | `pelvis` | `base_link` |
| 躯干 | `torso_link` | `torso_link` |
| 腰 | `waist_yaw_link` | `waist_yaw_link` |
| 右髋 | `right_hip_roll_link` | `r_hip_roll_link` |
| 右膝代理 | `right_knee_link` | `r_calf_link` |
| 右踝 | `right_ankle_roll_link` | `r_ankle_roll_link` |
| 左髋 | `left_hip_roll_link` | `l_hip_roll_link` |
| 左膝代理 | `left_knee_link` | `l_calf_link` |
| 左踝 | `left_ankle_roll_link` | `l_ankle_roll_link` |
| 右肩 | `right_shoulder_roll_link` | `r_shoulder_roll_link` |
| 右肘 | `right_elbow_link` | `r_elbow_link` |
| 右腕末端 | `right_wrist_yaw_link` | `r_wrist_link` |
| 左肩 | `left_shoulder_roll_link` | `l_shoulder_roll_link` |
| 左肘 | `left_elbow_link` | `l_elbow_link` |
| 左腕末端 | `left_wrist_yaw_link` | `l_wrist_link` |

### 1.4 物理参数对比

| 参数 | G1 腿部 | PiPlus 腿部 | G1 臂部 | PiPlus 臂部 |
|------|--------|------------|--------|------------|
| 力矩上限 | 88–139 Nm | 20 Nm | 5–35 Nm | 10 Nm |
| 速度上限 | 20–32 rad/s | 5.45 rad/s | 22–37 rad/s | 5.45–15.91 rad/s |
| 初始高度 | 0.76 m | **待测量** | — | — |
| PD Stiffness | 由电机常数计算 | **待测定** | 由电机常数计算 | **待测定** |
| PD Damping | 由电机常数计算 | **待测定** | 由电机常数计算 | **待测定** |

> PiPlus URDF 中未提供电机常数（armature），需从硬件规格书获取或通过系统辨识确定，才能复用 G1 的 PD 增益计算方式。

### 1.5 初始姿态差异

G1 的默认初始关节角度（`init_state.joint_pos`）：

```
hip_pitch: -0.312 rad
knee:       0.669 rad
ankle_pitch:-0.363 rad
elbow:      0.6 rad
left_shoulder_roll:  0.2 rad
left_shoulder_pitch: 0.2 rad
right_shoulder_roll: -0.2 rad
right_shoulder_pitch: 0.2 rad
```

PiPlus 需要根据其运动学重新标定默认站立姿态。


---

## 二、需要修改的文件清单

### 2.1 新建文件

| 文件路径 | 说明 |
|---------|------|
| `source/whole_body_tracking/soccer/robots/piplus.py` | PiPlus `ArticulationCfg` 定义，仿照 `g1.py` |
| `source/whole_body_tracking/soccer/tasks/tracking/config/piplus/` | PiPlus 专属配置目录 |
| `source/whole_body_tracking/soccer/tasks/tracking/config/piplus/__init__.py` | Gym 环境注册 |
| `source/whole_body_tracking/soccer/tasks/tracking/config/piplus/flat_env_cfg.py` | 基础平地环境配置 |
| `source/whole_body_tracking/soccer/tasks/tracking/config/piplus/soccer_flat_env_cfg.py` | 足球环境配置 |
| `source/whole_body_tracking/soccer/tasks/tracking/config/piplus/agents/rsl_rl_ppo_cfg.py` | PPO Runner 配置 |

### 2.2 需要修改的现有文件

| 文件路径 | 修改内容 |
|---------|----------|
| `source/whole_body_tracking/soccer/assets/` | 添加 PiPlus URDF/资产路径或将 `PiPlus/` 目录移入 |
| `source/whole_body_tracking/soccer/robots/__init__.py` | 导出 `PIPLUS_CFG` |

---

## 三、各文件详细适配说明

### 3.1 `robots/piplus.py` — ArticulationCfg

这是最核心的适配文件，需要完成以下内容：

**（1）URDF 路径**

```python
asset_path = f"{ASSET_DIR}/../../../PiPlus/urdf/PiPlus_S_12L8A0G2H1W.urdf"
# 或将 PiPlus/ 整体移入 assets/ 目录下
```

**（2）PD 增益**

G1 使用电机 armature 常数推导 PD 增益（`K = armature * ω²`，`D = 2ζ * armature * ω`）。
PiPlus 的 armature 未知，有两种方案：
- 方案 A：从 PiPlus 硬件规格书获取电机参数，复用相同公式
- 方案 B：直接填写经验值，例如参考 PiPlus XML 中的 `damping=0.02` 并从小值逐步调优

**（3）执行器分组**

G1 的 `actuators` 字段按关节类型分组（legs/feet/waist/arms）。PiPlus 需重新分组：

```python
actuators = {
    "legs": ImplicitActuatorCfg(
        joint_names_expr=[
            ".*_hip_pitch_joint", ".*_hip_roll_joint", ".*_thigh_joint",
        ],
        effort_limit_sim=20.0,
        velocity_limit_sim=5.45,
        stiffness=...,   # 待定
        damping=...,     # 待定
        armature=...,    # 待定
    ),
    "knees": ImplicitActuatorCfg(
        joint_names_expr=[".*_calf_joint"],
        effort_limit_sim=20.0,
        velocity_limit_sim=5.45,
        ...
    ),
    "feet": ImplicitActuatorCfg(
        joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
        effort_limit_sim=20.0,
        velocity_limit_sim=5.45,
        ...
    ),
    "waist": ImplicitActuatorCfg(
        joint_names_expr=["waist_yaw_joint"],
        ...
    ),
    "arms": ImplicitActuatorCfg(
        joint_names_expr=[
            ".*_shoulder_pitch_joint", ".*_shoulder_roll_joint",
            ".*_upper_arm_joint", ".*_elbow_joint",
        ],
        effort_limit_sim=10.0,
        velocity_limit_sim=15.91,
        ...
    ),
}
```

**（4）初始高度和关节角**

需要重新标定 PiPlus 站立时的 base_link 高度（G1 为 0.76 m）及默认关节角。

**（5）`init_state` 示例框架**

```python
init_state=ArticulationCfg.InitialStateCfg(
    pos=(0.0, 0.0, <PiPlus站立高度>),
    joint_pos={
        ".*_hip_pitch_joint": <待标定>,
        ".*_calf_joint": <待标定>,
        ".*_ankle_pitch_joint": <待标定>,
        ".*_elbow_joint": <待标定>,
    },
    joint_vel={".*": 0.0},
),
```


### 3.2 `config/piplus/flat_env_cfg.py` — 环境配置

仿照 `config/g1/flat_env_cfg.py`，需修改以下内容：

**（1）导入替换**

```python
# 旧
from soccer.robots.g1 import G1_ACTION_SCALE, G1_CYLINDER_CFG
# 新
from soccer.robots.piplus import PIPLUS_ACTION_SCALE, PIPLUS_CFG
```

**（2）机器人挂载**

```python
self.scene.robot = PIPLUS_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
self.actions.joint_pos.scale = PIPLUS_ACTION_SCALE
```

**（3）anchor body 名称**

G1 使用 `torso_link` 作为 anchor。PiPlus 同样有 `torso_link`，此项可保持不变。

```python
self.commands.motion.anchor_body_name = "torso_link"  # 不变
```

**（4）`body_names` 列表 — 最关键的适配项**

这是运动跟踪奖励的核心，必须对应 PiPlus 实际 link 名称：

```python
# G1 原始配置
self.commands.motion.body_names = [
    "pelvis",
    "left_hip_roll_link",  "left_knee_link",  "left_ankle_roll_link",
    "right_hip_roll_link", "right_knee_link", "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",  "left_elbow_link",  "left_wrist_yaw_link",
    "right_shoulder_roll_link", "right_elbow_link", "right_wrist_yaw_link",
]

# PiPlus 对应配置（需修改）
self.commands.motion.body_names = [
    "base_link",            # pelvis → base_link
    "l_hip_roll_link",      # left_hip_roll_link
    "l_calf_link",          # left_knee_link → l_calf_link
    "l_ankle_roll_link",    # left_ankle_roll_link
    "r_hip_roll_link",      # right_hip_roll_link
    "r_calf_link",          # right_knee_link → r_calf_link
    "r_ankle_roll_link",    # right_ankle_roll_link
    "torso_link",           # 不变
    "l_shoulder_roll_link", # left_shoulder_roll_link
    "l_elbow_link",         # left_elbow_link
    "l_wrist_link",         # left_wrist_yaw_link → l_wrist_link
    "r_shoulder_roll_link", # right_shoulder_roll_link
    "r_elbow_link",         # right_elbow_link
    "r_wrist_link",         # right_wrist_yaw_link → r_wrist_link
]
```

> 注意：如果参考运动数据（`.npz`）是从 G1 采集的，body 数量必须匹配。若 body_names 数量或顺序不一致，运动跟踪奖励将出错。

### 3.3 `tracking_env_cfg.py` — 接触传感器与奖励的 body 名称

文件 `tracking_env_cfg.py` 中有以下硬编码的 G1 body 名称，需适配 PiPlus：

**接触奖励白名单（允许接触的 link）：**

```python
# G1 原始（第 296-297 行）
"left_ankle_roll_link",
"right_ankle_roll_link",

# PiPlus 对应
"l_ankle_roll_link",
"r_ankle_roll_link",
```

**接触惩罚黑名单（不期望接触的 link 正则）：**

```python
# G1 原始（第 264 行）
r"^(?!left_ankle_roll_link$)(?!right_ankle_roll_link$)"
 r"(?!left_wrist_yaw_link$)(?!right_wrist_yaw_link$).+$"

# PiPlus 对应（去掉腕关节，改前缀）
r"^(?!l_ankle_roll_link$)(?!r_ankle_roll_link$).+$"
```

**身体倾倒终止条件（第 201 行）：**

```python
# G1 原始
"asset_cfg": SceneEntityCfg("robot", body_names="torso_link")

# PiPlus —— torso_link 存在，可保持不变
```


### 3.4 `config/piplus/soccer_flat_env_cfg.py` — 足球环境

仿照 `config/g1/soccer_flat_env_cfg.py`，主要变化：

- 导入 `PIPLUS_CFG` 和 `PIPLUS_ACTION_SCALE` 替换 G1
- 所有 `G1FlatEnvCfg` 基类替换为 `PiPlusFlatEnvCfg`
- 足球接触检测中涉及脚部 link 名称需更新（`left_ankle_roll_link` → `l_ankle_roll_link` 等）

### 3.5 `config/piplus/__init__.py` — Gym 环境注册

仿照 `config/g1/__init__.py`，注册以下环境 ID（按需选择）：

```python
gym.register(
    id="Tracking-Flat-PiPlus-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.PiPlusFlatEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PiPlusFlatPPORunnerCfg",
    },
)
```

### 3.6 `config/piplus/agents/rsl_rl_ppo_cfg.py` — PPO 配置

仿照 G1 的 PPO 配置，核心需适配的是 `num_actions`：
- G1: 29
- PiPlus: 23（无腕关节）

```python
@configclass
class PiPlusFlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    # ...
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
```

> `num_actions` 由 Isaac Lab 根据 `ActionsCfg` 自动推断，但观测维度（`num_obs`）需验证——因为 `joint_pos` 和 `joint_vel` 观测的维度会从 29 变为 23。

---

## 四、运动数据（`.npz`）适配

这是适配工作中**难度最高**的部分。

### 4.1 问题根源

当前 `motions/` 目录中的参考运动数据是为 G1 采集/重定向的，包含：
- `joint_pos` / `joint_vel`：29 维
- `body_pos_w` / `body_quat_w` 等：基于 G1 的 body 列表和数量

PiPlus 只有 23 个关节，且 body 名称不同，**直接使用 G1 的 `.npz` 数据会导致维度不匹配错误**。

### 4.2 适配方案

**方案 A（推荐）：重新采集/重定向运动数据**
- 使用动作捕捉重定向工具（如 PHC、Isaac Lab Motion Retargeting）将人体运动重定向到 PiPlus
- 生成新的 `.npz` 文件，字段与 G1 格式相同但维度为 23

**方案 B：关节子集映射**
- 从 G1 的 29 维 `joint_pos` 中提取与 PiPlus 对应的 23 维子集
- 需在 `MultiMotionLoader` 中添加关节名称映射逻辑
- 腕关节数据直接丢弃

**方案 C：保留 body 跟踪，放弃关节跟踪**
- 仅使用 `body_pos_w` / `body_quat_w`（世界坐标系下的 body 状态）做跟踪
- 跳过 `joint_pos` 的直接比较
- 适合在没有新运动数据时快速验证

### 4.3 `body_names` 与 `.npz` 的对齐要求

`commands_multi_motion.py` 中的 `MultiMotionLoader` 在加载时：
1. 按 `flat_env_cfg.py` 中 `body_names` 列表的顺序索引 `.npz` 中的 body 数据
2. body 数量必须与 `body_names` 长度完全一致
3. 关节数量必须与实际 articulation DOF 一致

因此必须保证：**新 `.npz` 文件的 body 顺序 = PiPlus `body_names` 列表顺序**。


---

## 五、MDP 模块适配

### 5.1 奖励函数（`mdp/rewards.py`）

奖励函数本身不依赖具体关节名，通过 `body_names` 索引 body 状态，适配后一般无需修改。
但以下项需要验证：

- `undesired_contacts`：正则黑名单中的 link 名称需更新（见 3.3）
- `motion_relative_body_position_error_exp` 等：依赖 `body_names` 数量和顺序，间接受影响

### 5.2 观测函数（`mdp/observations.py`）

- `joint_pos_rel` / `joint_vel_rel`：维度自动跟随 articulation DOF，29→23，无需手动改
- `robot_body_pos_b` / `robot_body_ori_b`：依赖 `body_names`，间接受影响
- `motion_anchor_pos_b`：anchor body 为 `torso_link`，PiPlus 有此 link，不变

### 5.3 终止条件（`mdp/terminations.py`）

- `bad_anchor_pos_z_only`：检测 `torso_link` 高度，PiPlus 有此 link，但阈值（z 高度下限）需重新标定
- `bad_anchor_ori`：检测躯干姿态，不依赖名称，无需修改
- `bad_motion_body_pos_z_only`：检测 body 列表中各 body 的高度，间接依赖 `body_names`

### 5.4 接触检测（`mdp/kick_detection.py`）

`KickContactTracker` 通过 contact sensor 历史判断踢球事件，contact sensor prim 路径为 `{ENV_REGEX_NS}/Robot/.*`（通配符），自动覆盖所有 link，无需修改。

但足球环境中的 `foot_target_distance` 等奖励若使用硬编码的脚部 link 名称，需检查并替换。

---

## 六、URDF 加载注意事项

### 6.1 Mesh 路径问题

PiPlus URDF 中 mesh 路径格式为：
```
package://PiPlus_S_12L8A0G2H1W/meshes/xxx.STL
```

Isaac Lab 使用 `UrdfFileCfg` 加载时，`package://` 协议需要 ROS package 环境，或通过以下方式解决：
- 将 `PiPlus/` 目录放入 `assets/` 下，并修改 URDF 中路径为相对路径
- 或使用 Isaac Lab 的 `fix_base=False` + 绝对路径
- 推荐：将 URDF 中所有 `package://PiPlus_S_12L8A0G2H1W/` 替换为相对路径 `../meshes/`

### 6.2 MJCF vs URDF

目前 G1 使用 `UrdfFileCfg` 加载 URDF。PiPlus 同时提供了 URDF 和 MJCF XML，建议：
- 优先使用 **URDF**（与现有流程一致）
- MJCF 可用于 MuJoCo 仿真对比验证

### 6.3 `replace_cylinders_with_capsules`

G1 配置中启用了 `replace_cylinders_with_capsules=True`。PiPlus URDF 的 collision geometry 已经手工定义为 box/cylinder 原语，建议保留此选项以改善物理稳定性。

### 6.4 自碰撞

G1 启用了 `enabled_self_collisions=True`。PiPlus 体型较小，建议同样启用，并在训练初期观察是否有自碰撞导致的不稳定。

---

## 七、适配工作优先级排序

按照「能跑起来 → 能训练 → 效果好」三个阶段：

### 阶段一：让环境能启动

1. 修复 URDF mesh 路径（`package://` → 相对路径）
2. 创建 `robots/piplus.py`，填写正确的关节名称和基本执行器参数（可先用经验值）
3. 创建 `config/piplus/flat_env_cfg.py`，更新 `body_names` 列表
4. 注册 Gym 环境 ID
5. 修复 `tracking_env_cfg.py` 中的 link 名称正则（接触奖励/惩罚）

### 阶段二：让训练能收敛

6. 重新采集或重定向 PiPlus 运动数据（`.npz`）
7. 标定初始站立高度和默认关节角
8. 测定并填写 PD 增益（stiffness / damping / armature）
9. 调整终止条件的高度阈值

### 阶段三：足球任务适配

10. 更新 `soccer_flat_env_cfg.py` 中所有 body 名称
11. 验证踢球接触检测逻辑
12. 调整球速、距离等超参数以适应 PiPlus 的尺寸和力量

---

## 八、快速参考：G1 → PiPlus 名称映射表

```
G1 关节/Link 名                  →  PiPlus 关节/Link 名
─────────────────────────────────────────────────────
pelvis                           →  base_link
left_hip_pitch_joint             →  l_hip_pitch_joint
left_hip_roll_joint              →  l_hip_roll_joint
left_hip_yaw_joint               →  l_thigh_joint
left_knee_joint                  →  l_calf_joint
left_ankle_pitch_joint           →  l_ankle_pitch_joint
left_ankle_roll_joint            →  l_ankle_roll_joint
left_hip_roll_link               →  l_hip_roll_link
left_knee_link                   →  l_calf_link
left_ankle_roll_link             →  l_ankle_roll_link
right_hip_pitch_joint            →  r_hip_pitch_joint
right_hip_roll_joint             →  r_hip_roll_joint
right_hip_yaw_joint              →  r_thigh_joint
right_knee_joint                 →  r_calf_joint
right_ankle_pitch_joint          →  r_ankle_pitch_joint
right_ankle_roll_joint           →  r_ankle_roll_joint
right_hip_roll_link              →  r_hip_roll_link
right_knee_link                  →  r_calf_link
right_ankle_roll_link            →  r_ankle_roll_link
left_shoulder_pitch_joint        →  l_shoulder_pitch_joint
left_shoulder_roll_joint         →  l_shoulder_roll_joint
left_shoulder_yaw_joint          →  l_upper_arm_joint
left_elbow_joint                 →  l_elbow_joint
left_wrist_roll_joint            →  (无，删除)
left_wrist_pitch_joint           →  (无，删除)
left_wrist_yaw_joint             →  (无，删除)
left_shoulder_roll_link          →  l_shoulder_roll_link
left_elbow_link                  →  l_elbow_link
left_wrist_yaw_link              →  l_wrist_link
right_shoulder_pitch_joint       →  r_shoulder_pitch_joint
right_shoulder_roll_joint        →  r_shoulder_roll_joint
right_shoulder_yaw_joint         →  r_upper_arm_joint
right_elbow_joint                →  r_elbow_joint
right_wrist_roll_joint           →  (无，删除)
right_wrist_pitch_joint          →  (无，删除)
right_wrist_yaw_joint            →  (无，删除)
right_shoulder_roll_link         →  r_shoulder_roll_link
right_elbow_link                 →  r_elbow_link
right_wrist_yaw_link             →  r_wrist_link
waist_yaw_joint                  →  waist_yaw_joint  (不变)
torso_link                       →  torso_link       (不变)
waist_yaw_link                   →  waist_yaw_link   (不变)
```






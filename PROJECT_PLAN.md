# Controller Learning 项目计划（v0.1）

> 状态：v0.1 架构与范围已确认，正在按 M0 → M8 实施（进度见 STATUS.md）
>
> GitHub 仓库：controller-learning（开发阶段 private，v0.1 完成后 public）
>
> Python 包：controller_learning
>
> 项目标题：Controller Learning — A GPU-Parallel Race Car Control Benchmark
>
> 许可证：MIT
> 参考项目：[learnsyslab/lsy_drone_racing](https://github.com/learnsyslab/lsy_drone_racing)

## 1. 项目定位

controller-learning 是一个面向控制学习、算法开发和统一评测的 GPU 并行汽车竞速 Challenge。

它的首要身份是：

> 以求职作品集为目的、以 Benchmark 为结构、以教程方式呈现的 Controller 学习平台。

项目不是 Controller Demo 的简单集合，也不是完整自动驾驶系统。核心产品是可复用的环境、任务、接口和评测协议；PID、MPC、PPO 是证明平台可用的教学示例，后续用户可以添加自己的 Controller。

公开英文简介：

> A GPU-parallel race car control benchmark with procedurally generated tracks, pluggable controllers, and reproducible evaluation.

项目主要回答：

- 同一个 Controller 能否在未见随机赛道上完成比赛？
- PID、MPC 和学习控制在完赛率、圈速、误差和实时性上有什么差异？
- 如何在统一 Observation、Action、车辆和测试集下公平比较 Controller？
- 如何使用数百到数千个 GPU world 训练 RL Controller，同时用相同 Challenge 正式评测？

## 2. v0.1 范围

### 2.1 必须完成

- MuJoCo MJCF 四轮汽车模型。
- MJX-Warp NVIDIA GPU 批量物理。
- 单环境 CarRacingEnv。
- 批量 VecCarRacingEnv。
- 1024 个并行随机赛道 world。
- Level 0 固定赛道。
- Level 1 程序化随机赛道。
- Controller 目录插件系统。
- PID、CasADi MPC、PyTorch PPO 示例。
- 统一评测、结果文件、训练日志和 2D replay。
- Linux、Python 3.11、Pixi 一键环境。
- CPU GitHub CI 和本地 GPU benchmark。
- 英文 README、教程、API 文档和 MIT License。

### 2.2 v0.1 非目标

- Level 2/3。
- MPCC。
- 摄像头、LiDAR、SLAM 和视觉 RL。
- 多车竞速。
- 真实车辆、ROS 和 sim-to-real。
- 完整悬架、Pacejka 高级轮胎和空气动力学。
- 真实三维随机道路 Mesh。
- 在线投稿、隐藏评测服务器和不可信代码沙箱。
- macOS、原生 Windows 和 WSL2 正式支持。
- 多个物理后端的通用抽象。

## 3. 从 Reference 继承的构建思路

lsy_drone_racing 的关键不是无人机，而是 Challenge 分层：

| Reference | controller-learning |
|---|---|
| Crazyflow | MuJoCo + MJX-Warp |
| RaceCoreEnv | CarRaceCoreEnv |
| VecDroneRaceEnv | VecCarRacingEnv |
| Gates | 虚拟 checkpoint |
| Level 配置 | Level 0/1 TOML |
| Controller 文件 | Controller 目录插件 |
| sim.py | 单次调试与 replay |
| evaluate.py | 多赛道统一评测 |

Reference 的有效原则：

1. 底层仿真只负责车辆如何运动。
2. Challenge 决定赛道、Observation、Reward、进度、终止和评测。
3. Controller 只能通过公开接口读取信息并产生动作。
4. 单环境用于开发，批量环境用于训练。
5. Level 通过配置定义，不在 Controller 中硬编码。
6. 正式评测统一加载 Controller 并重复运行。

本项目有两项有意改进：

- RL 必须直接训练正式 VecCarRacingEnv，不维护另一套简化训练环境。
- render callback 只获得写入式 DebugDraw，不获得 Simulator 内部对象。

本地 reference 目录仅用于研究，必须加入 .gitignore，不提交到公开仓库。

## 4. 核心系统模型

~~~mermaid
flowchart TB
    CFG["Level / Controller Config"] --> RUN["Runner"]
    TRACK["Track Generator and Pool"] --> ENV["CarRaceCoreEnv"]
    VEH["Four-wheel MJCF Vehicle"] --> PHYS["MJX-Warp Physics"]
    PHYS --> ENV
    ENV --> SINGLE["CarRacingEnv"]
    ENV --> VEC["VecCarRacingEnv"]
    SINGLE --> PLUGIN["PID / MPC / PPO Controller Plugin"]
    VEC --> PPO["GPU PPO Training"]
    PLUGIN --> EVAL["Evaluator"]
    EVAL --> ART["CSV / JSON / Plots / Replay"]
~~~

系统分为五个职责清楚的区域：

1. Physics：四轮车辆、执行器、轮地接触和状态推进。
2. Track：赛道几何、验证、池化和版本化。
3. Challenge：Observation、Action、Reward、进度和终止。
4. Controller：插件、示例算法和 PPO checkpoint。
5. Evaluation：正式协议、指标、结果和可视化。

## 5. 物理技术路线

### 5.1 正式后端

v0.1 使用：

- MuJoCo：MJCF 编译、车辆模型调试和 CPU 对照。
- MJX-Warp：Linux + NVIDIA 上的正式 GPU 仿真。
- JAX：批量状态、赛道几何、Reward、终止和 reset。

正式训练和正式 Controller 评测统一使用 MJX-Warp。CPU MuJoCo 只用于模型开发和短时一致性检查。

选择 MJX-Warp 的原因：

- 支持 MJX Model/Data batch dimension。
- 面向 NVIDIA GPU 优化。
- 相比 MJX-JAX 更适合持续轮地接触和约束。
- 可以用同一个 MJCF 模型运行单 world 和 1024 worlds。
- 技术路线最接近 reference 的 JAX/GPU 并行结构。

MetaDrive 不作为 v0.1 后端，因为它主要通过一进程一实例的 CPU 并行扩展，无法满足原生数百/数千 GPU world 的硬需求。

Isaac Lab 暂不采用，因为其系统更重、平台绑定更强，并且没有直接解决本项目的随机封闭汽车赛道问题。

### 5.2 GPU 并行验收

必须测试：

- num_envs = 1、64、256、1024。
- 1024 world 使用不同随机赛道。
- 每个 world 独立终止和 autoreset。
- 连续运行至少 10,000 个 environment steps。
- 无 NaN、接触缓冲溢出和持续显存增长。
- 输出编译时间、steps/s、transitions/s 和显存。

核心指标：

    transitions_per_second = num_envs × environment_steps_per_second

2048/4096 world 可以作为额外 benchmark，不是 v0.1 发布门槛。

### 5.3 Go / No-Go 与 fallback

MJX-Warp 四轮车辆是第一个技术门槛，必须先于完整赛道、Controller 和 RL 开发。

如果 1024-world 四轮接触失败，依次执行：

1. 调整 dt、solver、contact buffer 和摩擦参数。
2. 简化车轮关节和碰撞几何。
3. 保留公开 Environment API，切换为纯 JAX 平面四轮车辆动力学。

Fallback 仍按四个轮胎分别计算滑移与轮胎力，不把 bicycle model 作为仿真真值。GPU 原生批量是硬需求，不能退回只能 CPU 多进程的方案。

## 6. v0.1 四轮汽车

### 6.1 物理结构

汽车包含：

- 一个 6-DoF 刚性车身。
- 四个独立物理车轮。
- 四个车轮旋转关节。
- 两个前轮转向关节。
- 驱动和制动力矩。
- 轮地摩擦接触。
- 固定质量、惯量、轴距、轮距和车辆宽度。
- 转向角、转向速率和纵向力限制。

暂不包含：

- 独立悬架。
- 变速箱和复杂动力总成。
- 差速器细节。
- Pacejka 高级轮胎。
- 空气动力学。
- 路面高度变化。

### 6.2 仿真模型与 Controller 模型

必须区分：

- 仿真真值：四轮 MuJoCo/MJX-Warp 汽车。
- Controller 预测模型：Controller 内部可使用运动学或动力学简化模型。

MPC 使用简化汽车模型不等于仿真对象是自行车。复杂四轮 plant 与简化预测模型之间的误差是有意义的控制实验条件。

### 6.3 速度与赛道尺度

v0.1 定位为中等速度控制竞速，不追求 F1 高速动力学。

初始候选范围：

- 最高速度约 15 m/s。
- 赛道宽度约 6–8 m。
- 最小弯道半径约 12–15 m。
- 赛道长度约 300–600 m。
- 平坦地面与固定摩擦。

具体数值由车辆稳定性和 M1/M2 benchmark 锁定。

### 6.4 时序

Controller 频率候选为 20 Hz，所有 Controller 必须相同。

物理步长不提前锁定，M1/M2 比较：

- dt = 0.010 s（100 Hz）。
- dt = 0.005 s（200 Hz）。
- dt = 0.002 s（500 Hz）。

选择满足四轮接触稳定和 1024-world 性能要求的最大 dt。control_dt / physics_dt 必须是正整数。

## 7. 赛道系统

### 7.1 赛道真值

物理世界只包含统一平面和四轮汽车。每个 world 的赛道是独立 JAX 几何数据：

~~~python
class Track:
    seed
    centerline
    left_boundary
    right_boundary
    track_mask
    checkpoints
    start_pose
    length
    width
    generator_version
~~~

出界、进度、checkpoint 和 Reward 由这些数组计算。铺装路面只用于 replay 可视化，不进入物理碰撞。

### 7.2 固定容量表示

Gymnasium 和 GPU batch 要求固定 shape。

~~~text
centerline:     (max_track_points, 2)
left_boundary:  (max_track_points, 2)
right_boundary: (max_track_points, 2)
track_mask:     (max_track_points,)
checkpoints:    (max_checkpoints, ...)
~~~

规则：

- 使用固定弧长间距采样。
- 无效尾部填零并由 mask 标记。
- 超过容量的赛道在生成阶段拒绝。
- 不采用每条赛道相同点数但不同空间分辨率的方案。

max_track_points、采样间距和 max_checkpoints 由赛道尺度 spike 确定。

### 7.3 程序化生成

v0.1 使用：

1. 采样 8–16 个极角有序控制点。
2. 随机化半径与角度间隔。
3. 拟合周期三次样条。
4. 按固定弧长重新采样。
5. 计算切向、法向和曲率。
6. 偏移生成左右边界。
7. 生成起点和有序 checkpoint。
8. 运行几何与可驾驶性验证。

Level 1 只随机平面闭环几何。以下属性保持固定：

- 车辆参数。
- 起步状态。
- 赛道宽度。
- 地面摩擦。
- 物理平面。
- 障碍物数量为零。

### 7.4 几何验证

每条赛道必须满足：

- 中心线闭合且不自相交。
- 左右边界不自相交且不互相穿越。
- 非相邻路段保持最小间距。
- 曲率符合车辆最小转弯半径。
- 宽度与有效边界合法。
- 长度处于 v0.1 范围。
- 起点附近有足够直线。
- checkpoint 顺序和方向合法。
- 相同 seed + generator version 生成相同几何。

### 7.5 可驾驶性验证

几何合法后，候选赛道还必须通过正式四轮仿真中的低速参考 Controller：

- 沿中心线。
- 使用保守目标速度。
- 完成一圈。
- 不追求圈速。

无法完成的赛道不进入正式 pool，并保存失败原因。内部 validator 不参加 Controller 排名。

### 7.6 Train / Validation / Test

三者来自完全相同的生成分布，只使用不同 seed，不在 test 中秘密增加难度。

v0.1 使用离线生成的赛道池：

- Train：约 10,000 条，训练时上传 GPU 并在 reset 随机抽取。
- Validation：固定几何并提交仓库。
- Test：至少 20 条固定几何并提交仓库。

Validation/Test 文件保存 seed、全部几何、checkpoint、起点、长度、生成器版本和校验结果。生成器未来改变时发布新的 benchmark version，不改变旧测试几何。

## 8. Level 定义

v0.1 只实现两个 Level。

| Level | 车辆 | 赛道 | 初始状态 | 可见信息 | 用途 |
|---|---|---|---|---|---|
| Level 0 | 固定 | 固定 | 标准静止起步 | 完整状态 + 完整赛道 | 教学、调试、测试 |
| Level 1 | 固定 | 随机几何 | 标准静止起步 | 完整状态 + 完整赛道 | 正式 Benchmark |

Level 0 不作为主要排行榜，避免 Controller 针对单赛道硬编码。

Level 1 是 v0.1 正式评测任务，测试同分布未见赛道泛化。

Level 0/1 均不随机：

- 车辆参数。
- 车辆初始位置、航向和速度。
- 摩擦。
- 赛道宽度。
- Observation 噪声。

标准起步：

- 后轴中心位于中心线起点。
- 航向与中心线切线一致。
- 纵向/横向速度为零。
- 横摆角速度为零。
- 转向角为零。

## 9. Gymnasium 环境

### 9.1 单环境

正式 ID：

    ControllerLearning/CarRacing-v0

接口：

~~~python
obs, info = env.reset(seed=seed)
obs, reward, terminated, truncated, info = env.step(action)
env.render()
env.close()
~~~

### 9.2 批量环境

VecCarRacingEnv 使用领先 batch dimension：

~~~text
observation: (num_envs, ...)
action:      (num_envs, 2)
reward:      (num_envs,)
terminated:  (num_envs,)
truncated:   (num_envs,)
~~~

每个 world：

- 拥有独立车辆状态。
- 抽取独立训练赛道。
- 独立计算进度与 Reward。
- 独立终止。
- 使用 mask 独立 autoreset。

使用 Gymnasium NEXT_STEP autoreset 语义。JIT 后 reset/step 不因 world 状态或赛道 seed 重新编译。

### 9.3 单环境与批量职责

- PID/MPC/PPO checkpoint 正式评测使用单环境 Controller 插件接口。
- PPO 训练直接使用 VecCarRacingEnv 批量数组。
- 不为 1024 world 创建 1024 个 Python Controller 对象。
- 不维护独立的简化 RL environment。

## 10. Observation、Action 与坐标

### 10.1 坐标与单位

全部使用 SI 单位和二维右手约定：

- 世界 +x：基准前方。
- 世界 +y：左侧。
- yaw：从 +x 逆时针为正，单位 rad。
- 车体 +x：汽车前方。
- 车体 +y：汽车左侧。
- steering_angle > 0：左转。
- longitudinal_acceleration > 0：加速。

项目提供：

- world_to_body。
- body_to_world。
- wrap_angle。
- project_to_track。

### 10.2 Observation

每一步 Observation 包含该 Level 允许 Controller 看到的全部信息：

~~~text
position                 世界坐标 (2,)
yaw                      rad
velocity_body            纵向/横向速度 (2,)
yaw_rate                 rad/s
steering_angle           rad
track_progress           [0, 1]
centerline               (max_track_points, 2)
left_boundary            (max_track_points, 2)
right_boundary           (max_track_points, 2)
track_mask               (max_track_points,)
track_length             scalar
~~~

不直接提供：

- lateral_error。
- heading_error。
- target_speed。
- nearest_centerline_index。
- 未来状态。
- 物理内部对象。

Controller 使用公开几何工具计算控制误差。

### 10.3 Action

统一动作使用物理单位：

~~~text
action[0] = steering_angle             rad
action[1] = longitudinal_acceleration  m/s²
~~~

标准执行器层对所有 Controller 相同：

- 转向位置执行器与转向速率限制。
- 期望纵向力到驱动/制动力矩映射。
- 最大转角、加速度和制动力限制。

有限但越界的动作会被裁剪并记录 saturation。NaN、Inf、错误 shape 或不可转换 dtype 立即终止为 invalid_action。

## 11. Controller 插件

### 11.1 目录格式

~~~text
controllers/
├── template/
│   ├── controller.py
│   └── config.toml
├── pid/
│   ├── controller.py
│   └── config.toml
├── mpc/
│   ├── controller.py
│   ├── config.toml
│   └── helpers.py
└── ppo/
    ├── controller.py
    ├── config.toml
    └── assets/
        └── policy.pt
~~~

controller.py 必须导出唯一的 Controller 子类。动态加载路径为 Controller 目录，不限制复杂 Controller 只能写一个文件。

### 11.2 Reference 风格接口

~~~python
class Controller(ABC):
    def __init__(self, obs, info, config):
        ...

    @abstractmethod
    def compute_control(self, obs, info=None):
        ...

    def step_callback(
        self,
        action,
        obs,
        reward,
        terminated,
        truncated,
        info,
    ):
        ...

    def episode_callback(self):
        ...

    def render_callback(self, debug_draw):
        ...
~~~

规则：

- 每个 episode 创建全新的 Controller 实例。
- 正式评测禁止跨 episode 状态与学习。
- config 只在构造时传入。
- Observation 每步传入。
- Controller 只能使用 obs、受限 info 和只读 public config。
- Controller 不获得 Environment、MJX Data 或 Simulator 引用。

### 11.3 Challenge 配置与 Controller 配置

Challenge 配置：

    configs/levels/level0.toml
    configs/levels/level1.toml

Controller 配置：

    controllers/<name>/config.toml

Runner 组合两者并生成只读 public config。Controller 不能修改正式 Level、测试赛道、动作限制或评测协议。

### 11.4 Info 边界

Reset info 仅包含：

- episode_seed。
- controller_seed。
- track_id。
- benchmark_version。

Step info 只包含运行标识和终止状态。终止时增加：

- termination_reason。
- lap_completed。
- lap_time。

Evaluator 内部诊断数据不会在控制循环中作为捷径暴露。

### 11.5 Controller seed

Environment seed 与 Controller seed 分离并确定性派生。随机 Controller 使用 initial_info 中的 controller_seed，不依赖未设种子的全局 random 状态。

### 11.6 DebugDraw

render_callback 只获得写入式 DebugDraw：

- line。
- points。
- text。

Controller 可以绘制参考线、目标点和预测轨迹，但不能读取 Simulator 内部真值。正式 headless 评测不调用 render_callback。

### 11.7 信任边界

v0.1 只运行用户自己或仓库审核过的可信 Controller：

- 无 Docker 沙箱。
- 无文件、网络、CPU 或内存隔离。
- 不运行来源不可信的插件。
- 在线不可信代码执行属于未来独立系统。

所有 Controller 共享根 Pixi 环境。新增依赖必须更新 pyproject.toml 和 pixi.lock。

## 12. Episode、进度与终止

### 12.1 Episode

每个 episode：

1. 从标准起点静止起步。
2. 按顺序穿过全部虚拟 checkpoint。
3. 再次穿过起终点线。
4. 成功完成一圈。

暂不包含热身圈、多圈、中途重置和出界恢复。

### 12.2 Checkpoint

- Checkpoint 是横跨赛道的无碰撞虚拟线。
- 必须按顺序穿过。
- 漏过 checkpoint 后，后续 checkpoint 不计。
- Checkpoint 只用于环境完成判定，不提供额外控制捷径。
- track_progress 基于最后合法进度更新，不能跳到相邻空间中的非相邻赛段。

### 12.3 有效边界

v0.1 没有墙体、路锥障碍和其他车辆。

出界使用：

- 后轴中心作为车辆参考点。
- 原始边界向内收缩半个车宽和安全裕量。
- 参考点离开有效区域后立即 off_track。

v0.1 不使用 collision 指标或碰撞终止。失败原因：

- off_track。
- timeout。
- invalid_action。
- controller_error。
- controller_init_timeout。

### 12.4 Timeout

Timeout 随赛道长度变化：

    max_episode_time = max(60 s, track_length / 3 m/s)

它只防止停车或无限运行，不用于圈速奖励。

### 12.5 Gymnasium 语义

- terminated：success、off_track、invalid_action。
- truncated：episode timeout。
- controller_error/init_timeout：Runner 终止并记录，不属于物理环境内部状态。

## 13. Reward

核心 Environment 只提供简单稳定的基础 Reward：

- 正向归一化进度。
- 成功额外 +1。
- off_track 或 invalid_action 为 -1。

正式排名完全不使用 Reward。

PPO 训练可以通过标准 VectorRewardWrapper 做 Reward shaping，但必须使用同一个 VecCarRacingEnv。训练 Reward 配置与 checkpoint 一起保存。

## 14. 示例 Controller

### 14.1 PID

纵向：

    曲率速度规划 → 目标速度 → 速度 PID → 纵向加速度

横向：

    中心线投影 → 横向/航向误差 → 级联 PID/PD → 转向角

包含：

- 动作限幅。
- 积分限幅。
- Anti-windup。
- 可解释参数。
- 教程与图示。

验收：稳定完成 Level 0，并能不按赛道重新调参直接运行 Level 1。

### 14.2 MPC

v0.1 使用 CasADi + IPOPT：

- 内部模型：运动学汽车模型。
- 输入：转向角和纵向加速度。
- 预测时域候选 1–2 秒。
- 轨迹：中心线与曲率速度曲线。
- 约束：速度、转角、加速度和有效赛道边界。
- Warm start。

不在 v0.1 引入 acados。若无法满足软 50 ms deadline，依次缩短时域、优化 warm start，并评估线性化 MPC + OSQP。

验收：Level 0 稳定完赛；Level 1 validation 约 80% 完赛率，用于证明 Challenge 可解。

### 14.3 PPO

v0.1 只实现一个 PyTorch PPO：

- CleanRL 风格训练循环。
- 直接使用正式 VecCarRacingEnv。
- JAX/MJX-Warp arrays 通过 Gymnasium JaxToTorch wrapper 连接 PyTorch。
- 候选 num_envs 从 256 开始，目标 1024+。
- MLP policy/value 网络。
- State-based Observation。
- 车辆状态 + 车体坐标局部赛道预览。
- Action 为转向角与纵向加速度。

PPO 目的首先是证明 GPU 环境和训练管线有效：

- 稳定训练。
- 指标持续记录。
- 明显优于随机动作。
- 保存 checkpoint。
- checkpoint 包装成普通单环境 Controller。
- 能通过正式 Evaluator。

不设置必须达到 MPC 完赛率的发布门槛。

### 14.4 RL 日志

默认本地记录：

~~~text
runs/ppo/<run_id>/
├── config.toml
├── metrics.csv
├── TensorBoard events
├── checkpoint.pt
├── training_curve.png
└── manifest.json
~~~

记录 Reward、success rate、episode length、loss、entropy、learning rate、transitions/s、GPU memory 和 reset 数量。

Weights & Biases 仅作为可选功能，不要求账号和网络。

## 15. 评测协议

### 15.1 正式任务

- Level 0：开发和教学。
- Level 1：v0.1 正式 Benchmark。
- 公开固定 Test 几何：至少 20 条。
- 每个 Controller 使用完全相同的测试顺序。
- Controller seed 由 episode seed 确定性派生。
- 正式单 Controller 评测逐赛道执行，不使用批量 Controller API。

### 15.2 排名

不设计综合总分。

主要排序：

1. 完赛率降序。
2. 完赛率相同时，平均成功圈速升序。

同时报告：

- 平均速度。
- 横向误差 RMS/P95/Max。
- 动作饱和率。
- 转向与加速度平滑度。
- Controller 计算时间 P50/P95/P99。
- Deadline miss rate。
- 每个 track 的结果。
- 失败原因分布。

### 15.3 时间预算

候选限制：

- Controller 初始化上限 30 s。
- Controller 运行频率 20 Hz。
- compute_control soft deadline 50 ms。
- 实时合格：P99 不超过控制周期且 miss rate 不超过 1%。

初始化不计入圈速，但单独记录。

v0.1 运行可信 Controller，因此限制为软检查：

- 返回后判断是否超时。
- 异常被捕获并记录 traceback。
- 不使用子进程 watchdog 强制中断永久卡死。

### 15.4 结果文件

~~~text
results/<benchmark_version>/<controller>/<run_id>/
├── results.csv
├── summary.json
├── run_manifest.json
├── trajectory.png
├── telemetry.png
└── selected_replays/
~~~

run_manifest 至少保存：

- Git commit。
- Benchmark version。
- Python/Pixi lock。
- MuJoCo、MJX-Warp、JAX、CUDA、PyTorch 版本。
- CPU/GPU 和操作系统。
- Level/Controller 配置。
- Environment/Controller seeds。
- Test track IDs。

### 15.5 RL 公平性

- PPO 只使用 Train pool 调参和训练。
- Validation 用于 checkpoint/超参数选择。
- Test 仅生成最终结果。
- 报告环境步数、墙钟时间、GPU、训练 seed、搜索次数和 checkpoint 规则。
- 跨 Level、Oracle 或额外地图实验必须单独标注。

## 16. 可视化

v0.1 主要使用高质量 2D 俯视 replay：

- 铺装赛道区域。
- 左右边界和起终点。
- 四轮汽车图形。
- 实际与参考轨迹。
- MPC 预测轨迹。
- 速度、转向、进度和圈速。
- 失败原因。

MuJoCo 3D Viewer 只用于检查车辆关节、车轮接触和姿态。

正式 20-track 评测默认 headless。评测结束后选择成功/失败 track 单独 replay 和录制 MP4/GIF。

完整 3D 随机赛道是 Future Work，不阻塞 v0.1。

## 17. 建议仓库结构

~~~text
controller-learning/
├── README.md
├── LICENSE
├── pyproject.toml
├── pixi.lock
├── PROJECT_PLAN.md
├── .gitignore
├── configs/
│   ├── levels/
│   │   ├── level0.toml
│   │   └── level1.toml
│   ├── vehicle.toml
│   └── benchmark.toml
├── controller_learning/
│   ├── assets/
│   │   └── vehicle/
│   │       ├── car.xml
│   │       └── meshes/
│   ├── physics/
│   │   ├── model.py
│   │   ├── mjx_warp.py
│   │   ├── cpu_reference.py
│   │   └── jax_four_wheel_fallback.py
│   ├── tracks/
│   │   ├── types.py
│   │   ├── generator.py
│   │   ├── geometry.py
│   │   ├── validator.py
│   │   ├── pool.py
│   │   └── benchmark/
│   ├── envs/
│   │   ├── race_core.py
│   │   ├── car_racing.py
│   │   ├── vector_racing.py
│   │   ├── observation.py
│   │   ├── reward.py
│   │   └── termination.py
│   ├── control/
│   │   ├── base.py
│   │   ├── loader.py
│   │   └── debug_draw.py
│   ├── evaluation/
│   │   ├── evaluator.py
│   │   ├── metrics.py
│   │   ├── manifest.py
│   │   └── report.py
│   └── visualization/
│       ├── renderer_2d.py
│       └── replay.py
├── controllers/
│   ├── template/
│   ├── pid/
│   ├── mpc/
│   └── ppo/
├── scripts/
│   ├── sim.py
│   ├── evaluate.py
│   ├── generate_tracks.py
│   ├── validate_tracks.py
│   ├── benchmark_gpu.py
│   ├── train_ppo.py
│   └── replay.py
├── tests/
│   ├── unit/
│   ├── integration/
│   └── gpu/
├── benchmarks/
├── results/
└── docs/
~~~

## 18. Pixi、平台与 CI

### 18.1 v0.1 平台

- Linux x86-64。
- Python 3.11。
- NVIDIA GPU + CUDA 用于正式仿真和 RL。
- Pixi 是唯一正式安装方式。
- 不维护 Docker/Dev Container。

### 18.2 Pixi 环境

~~~text
default：CPU 开发、几何测试、MuJoCo 模型调试、文档
gpu：MJX-Warp、CUDA JAX、PyTorch、PPO、正式 benchmark
~~~

典型命令：

~~~bash
pixi install
pixi run tests
pixi run sim

pixi install -e gpu
pixi run -e gpu gpu-tests
pixi run -e gpu benchmark-gpu
pixi run -e gpu train-ppo
~~~

所有 Controller 共享根 Pixi 环境和 pixi.lock。

### 18.3 CI

GitHub Actions CPU CI：

- Pixi locked install。
- Ruff format/lint。
- Unit tests。
- Track generator/validator。
- Controller loader。
- Observation/Action schema。
- CPU MuJoCo model load。
- 短 CPU rollout。
- 文档 build。

GPU tests 在本地 NVIDIA 机器手动执行并提交版本化报告：

    benchmarks/v0.1/gpu_report.json

v0.1 不维护 self-hosted GPU runner。

## 19. 测试与验证

### 19.1 物理

- MJCF 能在 CPU MuJoCo 和 MJX-Warp 加载。
- 车辆静止稳定。
- 直行、转向、制动和动作限幅正确。
- 四轮接触无持续穿透或爆炸。
- CPU 与 MJX-Warp 短 rollout 在明确容差内一致。
- 100/200/500 Hz 候选测试。

### 19.2 Track

- Seed + generator version 确定性。
- 无自相交和非法边界。
- 曲率、宽度、长度和间距。
- Fixed-capacity padding/mask。
- checkpoint 顺序。
- 低速四轮可驾驶性。
- Train/validation/test 无 ID 或几何重复。

### 19.3 Environment

- Gymnasium checker。
- reset/step shape 与 dtype。
- 单环境和 batch=1 一致。
- Off-track 有效边界。
- 进度不跨非相邻赛段跳跃。
- success/terminated/truncated 语义。
- Masked autoreset 不改变其他 world。
- 固定 seed 的赛道和随机状态可复现。

### 19.4 Controller

- 每个目录只加载一个入口 Controller。
- PID/MPC/PPO action shape、dtype 和范围。
- 新实例无跨 episode 状态。
- Controller 异常与非法动作正确记录。
- DebugDraw 不暴露 Simulator。

### 19.5 GPU

- 1/64/256/1024 worlds。
- 1024 worlds × 10,000 steps。
- 不同赛道和独立 autoreset。
- 无 NaN/overflow/leak。
- Steps/s、transitions/s、显存和编译时间。
- PPO smoke training。

## 20. 实施里程碑

### M0：项目骨架

目标：

- 初始化 private Git 仓库。
- MIT License、.gitignore 和 reference 排除。
- Python 3.11 + Pixi default/gpu 环境。
- 包、测试、Ruff 和 CPU CI。
- 核心数据 schema 与配置加载。

验证：

- pixi install。
- pixi run tests。
- GitHub CPU CI 通过。

停止条件：

- MuJoCo/MJX-Warp/CUDA 依赖无法由 Pixi 锁定时，先解决环境，不进入物理开发。

### M1：CPU MuJoCo 四轮汽车

目标：

- 建立 MJCF 四轮汽车。
- 转向、驱动、制动和状态读取。
- CPU Viewer 调试。
- 扫描候选物理 dt。

验证：

- 静止、直行、转向、制动测试。
- 无接触爆炸。
- 状态单位与坐标约定正确。

停止条件：

- 简化结构仍无法稳定轮地接触时，先修正模型，不进入批量环境。

### M2：MJX-Warp GPU Go / No-Go

目标：

- 同一 MJCF 转入 MJX-Warp。
- JIT/vmap step。
- 1/64/256/1024 world benchmark。
- 调整 contact/constraint buffer。

验证：

- 1024 worlds 连续 10,000 steps。
- 无 NaN、overflow、显存泄漏。
- CPU/MJX-Warp 短 rollout 容差测试。
- GPU benchmark 报告。

停止条件：

- 如果调参和简化后仍失败，启动纯 JAX 四轮平面动力学 fallback。

### M3：批量 Track 与 Race Core

目标：

- Track 类型、周期样条 generator 和 validator。
- Fixed-capacity arrays/mask。
- Checkpoint、progress、effective boundary 和 timeout。
- Batched Reward、termination 和 masked reset。
- 低速可驾驶性验证。

验证：

- 批量几何 property tests。
- 不同 world 独立赛道与 reset。
- 进度和终止语义测试。

停止条件：

- Track shape 导致 JIT 重编译或 pool 无法装入 GPU 时，调整容量和采样分辨率。

### M4：Gymnasium 与 Controller Platform

目标：

- CarRacingEnv 和 VecCarRacingEnv。
- Gymnasium registration。
- Controller base/loader/template。
- Level/Controller 配置分离。
- DebugDraw 和单次 sim CLI。

验证：

- Gymnasium checker。
- Template Controller 完整跑通。
- Controller 无内部状态泄漏。

### M5：Level 0/1 与 Track Pool

目标：

- 固定 Level 0。
- 生成约 10,000 条 Train pool。
- 固定 Validation/Test 几何。
- 版本化 benchmark manifest。

验证：

- 所有正式 Track 通过几何与可驾驶性。
- Train/validation/test 分离。
- 1024 world 从 Train pool 快速 autoreset。

### M6：PID 与 MPC

目标：

- 纵向速度 PID 和横向级联 PID。
- CasADi + IPOPT 跟踪 MPC。
- 公共几何工具和曲率速度曲线。
- 文档与 DebugDraw。

验证：

- PID/MPC Level 0 完赛。
- MPC Level 1 validation 约 80% 完赛。
- 计算时间和 deadline 指标。

### M7：PPO 与 GPU 训练

目标：

- JaxToTorch wrapper。
- CleanRL 风格 PPO。
- Local CSV/TensorBoard logging。
- Checkpoint → Controller 插件。

验证：

- 1024-world smoke/full training。
- Policy 明显优于随机。
- Checkpoint 通过单环境 Evaluator。
- 训练 manifest 与 replay。

### M8：评测、作品展示与公开发布

目标：

- Level 1 正式 Test 评测。
- 结果表、轨迹、训练曲线和代表 replay。
- 英文 README、Quick Start、Controller tutorial、API docs。
- 清理 private 仓库并转 public。

验证：

- 新用户按 README 使用 Pixi 安装。
- 能运行 template/PID/MPC/PPO。
- 能复现已发布 benchmark。
- 没有密钥、reference 源码和未追踪大文件。

## 21. v0.1 完成定义

v0.1 只有同时满足以下条件才完成：

- Linux + Python 3.11 + Pixi 一键安装。
- 正式四轮汽车 GPU 后端稳定。
- 1024 个不同随机赛道 world 运行 10,000 steps。
- CarRacingEnv 和 VecCarRacingEnv。
- Level 0 和 Level 1。
- Controller 目录插件与模板。
- PID、MPC、PPO 示例。
- MPC Level 1 validation 约 80% 完赛。
- PPO 学习有效并导出可评测 checkpoint。
- 正式 Evaluator、结果文件和 2D replay。
- CPU CI 和版本化 GPU benchmark。
- 英文 README/API/教程与 MIT License。
- 仓库由 private 清理并转 public。

## 22. 主要风险

### 22.1 四轮接触吞吐不足

应对：

- M2 前置。
- 调整 solver/contact buffer/dt。
- 简化车轮结构。
- 必要时切换纯 JAX 四轮动力学。

### 22.2 随机赛道不可驾驶

应对：

- 几何硬约束。
- 低速四轮可驾驶性验证。
- 固定正式赛道几何。

### 22.3 RL 环境与正式环境分叉

应对：

- PPO 直接训练 VecCarRacingEnv。
- 只通过公开 wrapper 改变 Observation 特征和 Reward。

### 22.4 单环境与批量行为不同

应对：

- batch=1 一致性测试。
- 正式训练/评测均使用 MJX-Warp。

### 22.5 项目范围失控

应对：

- 只实现 Level 0/1。
- 只提供 PID/MPC/PPO。
- 2D 可视化优先。
- 每个 Milestone 有停止条件。

### 22.6 结果不可复现

应对：

- 固定 Pixi lock。
- 固定 Validation/Test 几何。
- 保存代码、依赖、硬件、配置和 seeds。
- 版本化 benchmark。

## 23. Future Work

v0.1 之后再考虑：

- Level 2：车辆参数随机化与隐藏真值。
- Level 3：局部/带噪赛道信息。
- MPCC。
- SAC/TD3 或其他 RL。
- 纯 JAX 在线赛道生成。
- 2048/4096 world 与多 GPU。
- macOS CPU/MJX-JAX 支持。
- Windows WSL2 支持。
- 完整 3D 随机道路。
- 多车竞速。
- ROS、真实车辆和 sim-to-real。
- 在线提交、隐藏 test 和安全沙箱。

## 24. 仍由实验锁定的参数

下列内容不是未解决的产品决策，而是 M1/M2/M3 必须测量的工程参数：

- 最终 physics_dt。
- MuJoCo solver 与 contact 参数。
- MJX-Warp naconmax/njmax。
- 驱动/制动力矩映射。
- max_track_points 与采样间距。
- 精确赛道长度、宽度和曲率范围。
- 单张 GPU 的最大稳定 world 数量。
- PPO 超参数。
- MPC 预测时域与权重。
- CPU/MJX-Warp 一致性容差。

这些参数必须通过 benchmark 和测试确定，不能只凭主观选择写死。

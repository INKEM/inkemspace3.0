---
title: 【项目学习】Unitree RL GYM (Go2)（04.01更新）
date: 2026-03-28
lastMod: 2026-04-01
summary: 基于 Unitree 机器人实现强化学习的示例仓库，以四足机器人 Go2 为例
tags: [宇树, IsaacGym, 四足机器人, PPO]
category: 项目学习
---

项目地址：https://github.com/unitreerobotics/unitree_rl_gym

标注说明：

- 正文
  - _配图/待补充_：博主后续更新才会在这里补充配图或正文。
  - ⚠️：需要注意的内容，否则可能会在后面遇到困惑。
  - 🐞：博主暂未解决的，可能是项目bug的问题，或者博主暂时无法解释的部分。
- 🔍代码定位（VSCode编辑器）
  - `...`：提示定位对应代码可以搜索的关键词。
  - 📌：提示定位对应代码可以右键使用“转到定义”的变量或函数，通常接续上一个展示的代码块。
  - `"..."`：提示定位对应代码可以右键使用“查找所有引用”的变量或函数，通常接续上一个展示的代码块。
  - `→`：需要通过多次操作中转才能定位对应代码。
  - 📃：给出代码段所在代码文件。
- 代码注释
  - `[*]...`：在必要的地方指引代码段在原始代码文件中的行数。
  - `(*)...`：将在代码块之后以正文形式补充讲解。
  - `(a, ..., b)`：一维张量的元素。
  - `[a, ..., b]`：n维张量的形状。

# 环境配置

参考文档：https://github.com/unitreerobotics/unitree_rl_gym/blob/main/doc/setup_zh.md

系统要求（博主本地配置）：

- 系统：Ubuntu 18.04 及以上（Ubuntu 22.04）
- 显卡：Nvidia 显卡（RTX 2050，4GB显存）
- 显卡驱动版本：525 及以上（580.126.09）
- Cuda 版本要求未提及，博主为 13.0

## 虚拟环境

下载并安装 MiniConda（清华镜像源）：

```bash
mkdir -p ~/miniconda3
wget https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
```

初始化 Conda：

```bash
~/miniconda3/bin/conda init --all
source ~/.bashrc
```

创建并激活虚拟环境`unitree-rl`：

```bash
conda create -n unitree-rl python=3.8
conda activate unitree-rl
```

后续终端操作均在`unitree-rl`虚拟环境下进行。

## 安装依赖

**安装 Pytorch**

Pytorch 是一个神经网络计算框架，用于模型训练和推理。

```bash
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia
```

**安装 Isaac Gym**

Isaac Gym 是 Nvidia 提供的刚体仿真和训练框架。

下载地址：https://developer.nvidia.com/isaac-gym/download

![|500](https://inkem-1306784622.cos.accelerate.myqcloud.com/blog/pic/Pasted%20image%2020260327201114.png)

解压后进入对应文件夹下的`isaacgym/python`目录进行安装：

```bash
cd isaacgym/python
pip install -e .
```

运行仿真示例验证安装结果：

```bash
cd examples
python 1080_balls_of_solitude.py
```

在 Ubuntu 22.04 下，Isaac Gym 会找不到 Python 3.8 的共享库文件，因此报错：

```
ImportError: libpython3.8.so.1.0: cannot open shared object file: No such file or directory
```

解决方法是设置环境变量。

临时解决方案（每次激活虚拟环境后输入）：

```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib
```

再次运行仿真示例即可成功：

![|500](https://inkem-1306784622.cos.accelerate.myqcloud.com/blog/pic/Pasted%20image%2020260327201645.png)

永久解决方案需要修改虚拟环境的激活脚本，在`unitree-rl`虚拟环境下：

```bash
# 修改激活脚本
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib' > $CONDA_PREFIX/etc/conda/activate.d/set_ld_library_path.sh
chmod +x $CONDA_PREFIX/etc/conda/activate.d/set_ld_library_path.sh
# 可选修改停用脚本，以防影响其他虚拟环境
mkdir -p $CONDA_PREFIX/etc/conda/deactivate.d
echo 'unset LD_LIBRARY_PATH' > $CONDA_PREFIX/etc/conda/deactivate.d/unset_ld_library_path.sh
chmod +x $CONDA_PREFIX/etc/conda/deactivate.d/unset_ld_library_path.sh
```

**安装 rsl_rl**

`rsl_rl`是一个强化学习算法库。

```bash
git clone https://github.com/leggedrobotics/rsl_rl.git
cd rsl_rl
# 切换版本分支以适配宇树项目
git checkout v1.0.2
pip install -e .
```

**安装 unitree_rl_gym**

`unitree_rl_gym`是项目本体。

```bash
git clone https://github.com/unitreerobotics/unitree_rl_gym.git
cd unitree_rl_gym
pip install -e .
```

## Auto DL

Auto DL是一个云算力租赁平台，若手头显卡性能较差可以考虑租赁。

**基础使用教程**

::bilibili{#BV1s2YdzNESZ}

Auto DL 的环境配置方法基本不变，本节以博主自身实践为例介绍在Auto DL上需要注意的几点。

博主的租赁配置：

- RTX4090 (24GB)×1；
- 基础镜像：PyTorch 2.3.0 & Python 3.12 (Ubuntu 22.04) & CUDA 12.1

Isaac Gym 安装包在本地下载，再拖到`/autodl-tmp`下，运行解压命令：

```bash
tar -zxvf 文件名.tar.gz
```

git 克隆如果遇到报错：

```text
fatal: unable to access 'https://github.com/leggedrobotics/rsl_rl.git/': GnuTLS recv error (-110): The TLS connection was non-properly terminated.
```

解决方法：

```bash
# 关闭 GnuTLS 的 SSL 验证（临时解决连接问题）
git config --global http.sslVerify false
```

解决一段时间后博主又遇到报错：

```text
fatal: unable to access 'https://github.com/unitreerobotics/unitree_rl_gym.git/': Failed to connect to github.com port 443 after 129504 ms: Connection timed out
```

无奈只能本地在项目网站上直接下载 zip ，再拖到`/autodl-tmp`下，运行解压命令：

```bash
unzip 文件名.zip
```

训练时需要使用有卡模式开机。

后续学习过程在 Auto DL 上的区别都会在相应步骤之后给出。

# 试运行

运行以下命令对`go2`四足机器人进行训练：

```bash
python legged_gym/scripts/train.py --task=go2
```

Auto DL 由于系统无图形化界面，训练需要使用`--headless`参数禁用图形渲染，这也能大大提高训练效率：

```bash
python legged_gym/scripts/train.py --task=go2 --headless
```

如果显卡性能不够，会有CUDA内存分配失败报错：

```
RuntimeError: The following operation failed in the TorchScript interpreter.
Traceback of TorchScript (most recent call last):
  File "/home/inkem/isaacgym/python/isaacgym/torch_utils.py", line 79, in quat_rotate_inverse
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * \
        torch.bmm(q_vec.view(shape[0], 1, 3), v.view(
        ~~~~~~~~~ <--- HERE
            shape[0], 3, 1)).squeeze(-1) * 2.0
    return a - b + c
RuntimeError: CUDA error: CUBLAS_STATUS_ALLOC_FAILED when calling `cublasCreate(handle)`
```

此时可以尝试减少并行训练的环境数量：

```python
# 🔍 num_envs
# 📃 legged_robot_config.py
class LeggedRobotCfg(BaseConfig):
	class env:
		# [5]训练环境数量由4096改成64
		num_envs = 64
		num_observations = 48
		num_privileged_obs = None
		num_actions = 12
		env_spacing = 3.
		send_timeouts = True
		episode_length_s = 20
		test = False
```

也可以在命令中使用`--num_envs`临时配置训练环境数量：

```bash
python legged_gym/scripts/train.py --task=go2 --num_envs=64
```

运行成功后，如果没有禁用图形渲染，将跳出图形界面显示训练过程：

![|500](https://inkem-1306784622.cos.accelerate.myqcloud.com/blog/pic/Pasted%20image%2020260327204950.png)

运行训练的终端将对每个回合更新显示训练数据，以下是博主第一轮训练倒数第二个回合的训练数据：

```
回合迭代进度
Learning iteration 1499/1500
仿真速度(环境交互收集数据耗时, 网络梯度更新耗时)
Computation: 1073 steps/s (collection: 1.289s, learning 0.141s)
价值函数损失，更新Critic网络
Value function loss: 0.0138
替代损失，更新Actor网络
Surrogate loss: -0.0102
动作噪声标准差
Mean action noise std: 1.13
平均总奖励
Mean reward: 0.09
平均回合长度
Mean episode length: 463.55
动作变化率惩罚
Mean episode rew_action_rate: -0.3149
水平方向角速度惩罚
Mean episode rew_ang_vel_xy: -0.3814
碰撞惩罚
Mean episode rew_collision: -0.0315
关节加速度惩罚
Mean episode rew_dof_acc: -0.3668
关节限位惩罚
Mean episode rew_dof_pos_limits: -0.1940
足端腾空时间惩罚
Mean episode rew_feet_air_time: -0.0755
垂直方向速度惩罚
Mean episode rew_lin_vel_z: -0.0611
扭矩大小惩罚
Mean episode rew_torques: -0.0851
角速度跟踪误差奖励
Mean episode rew_tracking_ang_vel: 0.1258
线速度跟踪误差奖励
Mean episode rew_tracking_lin_vel: 0.1891
--------------------------------------------------------------------------------
总时间步
Total timesteps: 2304000
上一回合长度
Iteration time: 1.43s
已训练时间
Total time: 2476.81s
预计剩余训练时间
ETA: 1.7s
```

先不详细解释各个奖励项，直接运行`play`命令看看训练效果：

```bash
python legged_gym/scripts/play.py --task=go2
```

博主的 RTX 2050 只能跑 64 个环境，一轮训练几乎看不到效果，机器人要么向后撑着，要么向前蛄蛹一下趴地上：

![|500](https://inkem-1306784622.cos.accelerate.myqcloud.com/blog/pic/Pasted%20image%2020260328194141.png)

而 Auto DL 上可以跑满 4096 个环境，一轮训练效果明显，就是机器人是斜着走的：

<video controls width="800" preload="metadata">
  <source src="https://inkem-1306784622.cos.accelerate.myqcloud.com/blog/pic/output.mp4" type="video/mp4">
  您的浏览器不支持视频播放。
</video>

Auto DL 无图形化界面，训练效果的查看方法详见**训练-训练效果**一节。

# 训练操作

接下来我们引入更多的训练参数，实现更灵活的训练过程。

官方文档给出的参数说明如下：

- `--task`：必选参数，值可选（go2, g1, h1, h1_2）；
- `--headless`：默认启动图形界面，设为 true 时不渲染图形界面（效率更高）；
- `--resume`：从日志中选择 checkpoint 继续训练；
- `--experiment_name`：运行/加载的 experiment 名称；
- `--run_name`：运行/加载的 run 名称；
- `--load_run`：加载运行的名称，默认加载最后一次运行；
- `--checkpoint`：checkpoint 编号，默认加载最新一次文件；
- `--num_envs`：并行训练的环境个数；
- `--seed`：随机种子；
- `--max_iterations`：训练的最大迭代回合数；
- `--sim_device`：仿真计算设备，指定 CPU 为 `--sim_device=cpu`；
- `--rl_device`：强化学习计算设备，指定 CPU 为 `--rl_device=cpu`。

## 训练的保存与加载

每次运行训练时，默认情况下，训练过程都会保存在`logs/<experiment_name>/<date_time>_<run_name>/model_<checkpoint>.pt`，其中

- `experiment_name`：实验名称，go2 机器人默认为`rough_go2`，通过`--experiment_name`可以自定义实验名称；
  - 用于定义实验主题，例如区分不同的训练任务（平地行走/崎岖地形/爬楼梯）。
- `date_time`：训练开始的时间，例如3月29日16时06分46秒为`Mar29_16-06-46`；
- `run_name`：运行名称，默认为空，通过`--run_name`可以给定运行名称；
  - 在每个实验主题下，区分不同超参数或配置的训练尝试。
- `checkpoint`：检查点，训练过程每 50 回合保存一次模型参数。

启动一次新训练时，我们可以通过`--experiment_name`和`--run_name`区分这次训练：

```bash
python legged_gym/scripts/train.py --task=go2 --experiment_name=exp1 --run_name=run1
```

从某一次训练中的模型参数加载时：

- `--resume`：声明训练从检查点加载；
- `--experiment_name`：指定训练所在的实验文件夹；
- `--load_run`：指定具体的训练，需要给出完整的`<date_time>_<run_name>`；
- `--checkpoint`：指定模型检查点；
- `--run_name`：不会继承，需要重新命名。

```bash
python legged_gym/scripts/train.py --task=go2 --resume --experiment_name=exp1 --load_run=Mar29_16-46-08_run1 --checkpoint=500 --run_name=run2
```

不指定`--load_run`和`--checkpoint`则默认加载最新的一次训练及训练中最新的模型。

> 🐞
>
> - 如果所加载的训练没有迭代到最大回合数就停止（例如`Ctrl+C`中断），则无论是否保存过模型参数，加载后的训练对已经迭代的回合数不会保留。
> - 在已经迭代的回合数被正常保留的情况下，终端显示的预计剩余时间`ETA`会变为负数。

## 训练的配置

部分配置可以通过命令行参数临时调整：

- `--num_envs`：并行训练的环境个数；
- `--seed`：训练使用的随机数种子；
- `--max_iteration`：单次训练的最大迭代回合数。

其他配置，例如奖励缩放因子和PPO超参数等，则需在代码中修改后再启动训练。

## 训练的设备

- `--sim_device`：仿真计算设备，默认为GPU，指定 CPU 为 `--sim_device=cpu`；
- `--rl_device`：强化学习计算设备，默认为GPU，指定 CPU 为 `--rl_device=cpu`。

## 训练效果

运行`play.py`可查看训练效果：

```bash
python legged_gym/scripts/play.py --task=go2
```

指定模型的参数包括`--experiment` `--load_run` `--checkpoint`，用法与运行训练的命令相同。

由此，Auto DL 的模型则可以通过下载到本地，并存放于与 Auto DL 中一致的路径下后，在本地运行训练效果。

## 训练可视化

TensorBoard 是 TensorFlow 提供的可视化工具包，用于机器学习实验。在`unitree-rl`虚拟环境中使用如下命令安装：

```bash
conda install tensorboard
```

于仓库文件夹`\unitree_rl_gym`下打开终端，运行如下命令可以使用 TensorBoard 工具将日志文件夹中的训练过程可视化：

```bash
tensorboard --logdir=logs
```

进入终端输出的网址：

```text
TensorBoard 2.20.0 at http://localhost:6006/ (Press CTRL+C to quit)
```

![](https://inkem-1306784622.cos.accelerate.myqcloud.com/blog/pic/Pasted%20image%2020260329171203.png)

左侧边栏勾选要可视化的训练，中间的分类展示了不同数据指标：

- **Episode**：机器人的具体表现，包括各个奖励项；
- **Loss**：神经网络的学习过程，包括学习率和损失函数；
- **Perf**：训练过程的运行效率，包括数据收集和网络更新的占用时间和环境模拟帧数；
- **Policy**：策略输出指标，包括输出动作的标准差；
- **Train**：训练的综合效果，包括平均总奖励和平均回合长度。

---

Auto DL 的自定义服务功能可以在本地访问租赁主机的 6006 和 6008 端口，TensorBoard 默认在 6006 端口运行，也可通过`--port`参数配置：

```bash
tensorboard --logdir=logs --port=6006
```

![](https://inkem-1306784622.cos.accelerate.myqcloud.com/blog/pic/Pasted%20image%2020260329212855.png)

# 奖励

`rewards`类定义了所有与奖励相关的配置参数：

```python
# 🔍 reward
# 📃 legged_robot_config.py
# [101]
class rewards:
	# 所有奖励函数的缩放因子
    class scales:
        termination = -0.0
        tracking_lin_vel = 1.0
        tracking_ang_vel = 0.5
        lin_vel_z = -2.0
        ang_vel_xy = -0.05
        orientation = -0.
        torques = -0.00001
        dof_vel = -0.
        dof_acc = -2.5e-7
        base_height = -0.
        feet_air_time =  1.0
        collision = -1.
        feet_stumble = -0.0
        action_rate = -0.01
        stand_still = -0.

    only_positive_rewards = True # if true negative total rewards are clipped at zero (avoids early termination problems)
    tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)
    soft_dof_pos_limit = 1. # percentage of urdf limits, values above this limit are penalized
    soft_dof_vel_limit = 1.
    soft_torque_limit = 1.
    base_height_target = 1.
    max_contact_force = 100. # forces above this value are penalized
```

定义了各奖励项计算方法的奖励函数：

```python
# 🔍 reward/任意奖励函数缩放因子名
# 📃 legged_robot.py
# [636]
#------------ reward functions----------------
def _reward_lin_vel_z(self):
    # Penalize z axis base linear velocity
    return torch.square(self.base_lin_vel[:, 2])

def _reward_ang_vel_xy(self):
    # Penalize xy axes base angular velocity
    return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)

def _reward_orientation(self):
    # Penalize non flat base orientation
    return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

# ...
```

⚠️ 所有的奖励函数都返回一个形状为`[num_envs（环境数量）]`的一维张量，存储每个环境各自结算的奖励项。

⚠️ 可以发现，`legged_robot.py`定义了强化学习环境具体的运行逻辑，而`legged_robot_config.py`是调整`legged_robot.py`运行参数的配置接口，是所有机器人训练的通用模板。`\envs`目录下分目录存储了所有机器人型号各自的`env`和`config`文件，它们继承自`legged_robot`文件，为训练专门的机器人型号做了特定的适配。`go2`的环境直接使用`legged_robot.py`，无需单独的`env`文件。

## 奖励函数

奖励在数学上默认包含正负两种情况，而奖励项的作用具体是激励还是惩罚在代码中由`scales`类处理，函数内部一概返回正奖励值。为了便于理解，对于给予负缩放因子的奖励项，博主将直接称呼其为惩罚以做区分。

### 任务目标奖励

```python
# [693]
def _reward_tracking_lin_vel(self):
    # Tracking of linear velocity commands (xy axes)
    lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
    return torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma)
```

**线速度跟踪奖励**奖励机器人基座在$XY$平面的实际线速度$\mathbf v_{b,xy}$跟踪指令线速度$\mathbf v^*_{b,xy}$的表现，提升足式机器人的平移能力。

$$
\exp(-\frac{\|\mathbf v^*_{b,xy}-\mathbf v_{b,xy}\|^2}\sigma)
$$

$\sigma$决定了奖励如何随跟踪误差衰减，对应参数`tracking_sigma`。$\sigma$越小，奖励曲线越陡峭，微小误差越能大幅降低奖励，相当于对跟踪误差的容忍度降低了。

_配图_

```python
# [698]
def _reward_tracking_ang_vel(self):
    # Tracking of angular velocity commands (yaw)
    ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
    return torch.exp(-ang_vel_error/self.cfg.rewards.tracking_sigma)
```

**角速度跟踪奖励**奖励机器人基座绕$Z$轴的实际角速度$\boldsymbol\omega_{b,z}$跟踪指令角速度$\boldsymbol\omega^*_{b,z}$的表现，提升足式机器人的转向能力。

$$
\exp(-\frac{\|\boldsymbol\omega^*_{b,z}-\boldsymbol\omega_{b,z}\|^2}\sigma)
$$

### 运动稳定性与能耗惩罚

```python
# [637]
def _reward_lin_vel_z(self):
    # Penalize z axis base linear velocity
    return torch.square(self.base_lin_vel[:, 2])
```

**线速度惩罚**$-\mathbf v_{b,z}^2$惩罚机器人基座在$Z$轴方向的线速度，保持基座高度稳定，抑制不必要的跳跃或下沉。

```python
# [641]
def _reward_ang_vel_xy(self):
    # Penalize xy axes base angular velocity
    return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
```

**角速度惩罚**$-||\boldsymbol\omega_{b,xy}||^2$惩罚机器人基座绕$X$轴和$Y$轴的角速度，保持基座姿态稳定，抑制侧翻与前后倾。

```python
# [645]
def _reward_orientation(self):
    # Penalize non flat base orientation
    return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)
```

**姿态惩罚**$-\|\mathbf g_{\mathrm{project}}\|^2$惩罚机器人重力向量在基座坐标系$XY$维度下的投影。相比角速度惩罚，姿态惩罚直接惩罚基座姿态倾斜的角度。

_配图_

同时惩罚角速度和角度是控制理论中 PD 控制在奖励函数中的体现，能够让机器人快速回正且抑制震荡。

```python
# [649]
def _reward_base_height(self):
    # Penalize base height away from target
    base_height = self.root_states[:, 2]
    return torch.square(base_height - self.cfg.rewards.base_height_target)
```

**高度惩罚**$-(h_b-h^*_b)^2$惩罚机器人基座高度偏离目标值，期望基座保持在理想的工作高度，因具体任务和训练效果选择性设置。

```python
# [654]
def _reward_torques(self):
    # Penalize torques
    return torch.sum(torch.square(self.torques), dim=1)
```

**力矩惩罚**$-\|\boldsymbol\tau\|^2$惩罚关节力矩过大，鼓励节能和低功耗运动。

```python
# [658]
def _reward_dof_vel(self):
    # Penalize dof velocities
    return torch.sum(torch.square(self.dof_vel), dim=1)
```

**速度惩罚**$-\|\dot{\mathbf p}\|^2$惩罚关节速度过大，鼓励平缓运动。

```python
# [662]
def _reward_dof_acc(self):
    # Penalize dof accelerations
    return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)
```

**加速度惩罚**$-\|(\dot{\mathbf p}_{t-\Delta t}-\dot{\mathbf p}_t)/\Delta t\|^2$惩罚关节加速度（通过相邻仿真时间步的关节速度差计算）过大，进一步鼓励平缓运动，减少冲击。

力矩惩罚和加速度惩罚具有相关性，但力矩惩罚还考虑了静载荷（重力）与摩擦等因素，更注重减少过载与能耗。

```python
# [666]
def _reward_action_rate(self):
    # Penalize changes in actions
    return torch.sum(torch.square(self.last_actions - self.actions), dim=1)
```

**动作变化率惩罚**$-\|\mathbf a_{t-1}-\mathbf a_t\|^2$从神经网络层面，惩罚其相邻决策时间步输出的动作空间指令。

动作空间往往是关节位置空间，因此动作变化率惩罚和速度、加速度惩罚也具有相关性。但出于计算性能的限制，神经网络的决策时间步通常大于执行器的执行时间步（仿真时间步），故执行器具体的执行效果在神经网络的观测中存在盲区，加速度惩罚则从更细的粒度上弥补了这一点。

除此之外，动作变化率惩罚通过鼓励神经网络学到平滑的策略，避免过大的梯度，对训练稳定性也起到重要作用。

### 物理安全与限制惩罚

```python
# [670]
def _reward_collision(self):
    # Penalize collisions on selected bodies
    return torch.sum(1.*(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)
```

**碰撞惩罚**惩罚非足端部位的碰撞次数。

⚠️碰撞次数的统计通过计算接触力实现：

1. `self.contact_forces[]`：一个三维张量`[num_envs, num_bodies(机器人刚体数量), 3]`，存储每个环境（机器人）的每个刚体的接触力在$XYZ$三个方向的分量；
2. `self.penalised_contact_indices[]`：存储并筛选所有需要惩罚碰撞的刚体索引；
3. `torch.norm()`：计算接触力的大小；
4. `> 0.1`：判断是否接触，阈值`0.1`排除传感器噪声和数值误差等影响；
5. `1.`：将布尔值`True`和`False`转化为`1`和`0`；
6. `torch.sum()`：对每个环境中判定为碰撞的刚体数量分别求和。

```python
# [674]
def _reward_termination(self):
    # Terminal reward / penalty
    return self.reset_buf * ~self.time_out_buf
```

**终止惩罚**惩罚环境触发提前终止的次数，具体机制与终止条件的判断函数`check_termination()`有关：

```python
# 🔍 "reset_buf"/"time_out_buf"
# 📃 legged_robot.py
# [118]
def check_termination(self):
    """ Check if environments need to be reset
    """
    self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)
    self.reset_buf |= torch.logical_or(torch.abs(self.rpy[:,1])>1.0, torch.abs(self.rpy[:,0])>0.8)
    self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
    self.reset_buf |= self.time_out_buf
```

对每个环境，布尔张量`self.reset_buf`判断其是否触发任意终止条件，`time_out_buf`判断其是否触发超时终止条件（到达回合时间限制）。

除了超时，终止条件还有：

- 接触力超过`1`的严重碰撞。
- 姿态严重倾斜（前倾/后仰超过约57°或侧倾超过约46°）。
  - `self.rpy`形状为`[num_envs, 3]`，以弧度制存储每个环境机器人的**横滚角**（Roll）、**俯仰角**（Pitch）和**转向角**（Yaw）。

_配图_

```python
# [678]
def _reward_dof_pos_limits(self):
    # Penalize dof positions too close to the limit
    out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.) # lower limit
    out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
    return torch.sum(out_of_limits, dim=1)
```

**关节限位惩罚**惩罚关节位置超出软限位的大小。

软限位是软件设置的限位，比物理硬件限位更严格，以留出惩罚作用幅度的余量。

```python
# [684]
def _reward_dof_vel_limits(self):
    # Penalize dof velocities too close to the limit
    # clip to max error = 1 rad/s per joint to avoid huge penalties
    return torch.sum((torch.abs(self.dof_vel) - self.dof_vel_limits*self.cfg.rewards.soft_dof_vel_limit).clip(min=0., max=1.), dim=1)
```

**关节速度限制惩罚**惩罚关节速度超出软限位的大小，但单个关节的最大惩罚截断为1。

因为位置（以及后文的力矩）在仿真器和硬件处理中都自带截断逻辑，但速度理论上可以无限增加（仿真器中），故需要在奖励计算时截断。过大的速度会导致仿真策略应用到实物因为摩擦力和空气阻力等而失效，同时避免速度惩罚过大，淹没其他奖励信号，甚至产生数值爆炸。

```python
# [689]
def _reward_torque_limits(self):
    # penalize torques too close to the limit
    return torch.sum((torch.abs(self.torques) - self.torque_limits*self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)
```

**关节力矩限制惩罚**惩罚关节力矩超出软限位的大小。

在速度惩罚和力矩惩罚的基础上，速度限制惩罚和力矩限制惩罚的作用在于进一步平衡奖惩机制，防止策略为了更高的任务奖励而牺牲安全性，对于超出安全边界的情况加大惩罚力度。

### 步态与行为风格

```python
# [703]
def _reward_feet_air_time(self):
    # Reward long steps
    # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
    # 通过垂向接触力与1比较判定每个足端是否接触
    contact = self.contact_forces[:, self.feet_indices, 2] > 1.
    # 滤波操作，仅当连续两个时间步均未接触才判定为腾空
    contact_filt = torch.logical_or(contact, self.last_contacts)
    self.last_contacts = contact
    # 判定足端是否产生首次接触，即腾空时间未重置且当前处于接触状态
    first_contact = (self.feet_air_time > 0.) * contact_filt
    # 累积腾空时间
    self.feet_air_time += self.dt
    # 将产生首次接触的足端的腾空时间折算为奖励，要求腾空时间至少0.5秒
    rew_airTime = torch.sum((self.feet_air_time - 0.5) * first_contact, dim=1) # reward only on first contact with the ground
    # 指令线速度超过0.1时才给予奖励，否则机器人在接收静止指令时会被鼓励原地起跳
    rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1 #no reward for zero command
    # 接触足端腾空时间重置
    self.feet_air_time *= ~contact_filt
    return rew_airTime
```

**足端腾空时间奖励**鼓励机器人达到小跑乃至奔跑的步态，通过足端从起跳到落地的时间间隔来折算。通常小跑和奔跑的腾空时间分别在0.3～0.5秒和0.5～0.8秒。

```python
# [716]
def _reward_stumble(self):
    # Penalize feet hitting vertical surfaces
    return torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) >\
        5 *torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)
```

**绊倒惩罚**惩罚足端与垂直表面的接触，以足端受到的侧向力为判断依据，当其超过垂向力的5倍时，我们可以认为这是异常撞击而非正常推进（摩擦力）。

```python
# [721]
def _reward_stand_still(self):
    # Penalize motion at zero commands
    return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1) * (torch.norm(self.commands[:, :2], dim=1) < 0.1)
```

**站立姿态惩罚**是一个仅在命令机器人静止时（指令线速度小于0.1）被计算的条件奖励，鼓励机器人保持站立姿态。`📌default_dof_pos`是预设站立姿态下各关节位置，惩罚由当前各关节位置与预设位置之间的误差和计算而来。

```python
# [725]
def _reward_feet_contact_forces(self):
    # penalize high contact forces
    return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) - self.cfg.rewards.max_contact_force).clip(min=0.), dim=1)
```

**足端接触力限制惩罚**惩罚足端接触力超过预设最大值的部分，鼓励轻柔触地。

## 奖励的统一处理

`_prepare_reward_function()`是奖励的初始化函数，用于动态构建奖励函数列表并进行预处理：

```python
# 🔍 reward
# 📃 legged_robot.py
# [485]
def _prepare_reward_function(self):
	""" Prepares a list of reward functions, whcih will be called to compute the total reward.
		Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
	"""
	# remove zero scales + multiply non-zero ones by dt
	# 遍历奖励项缩放因子列表
	for key in list(self.reward_scales.keys()):
		scale = self.reward_scales[key]
		if scale==0:
			# 过滤零缩放因子项，节省计算
			self.reward_scales.pop(key)
		else:
			# (1)
			self.reward_scales[key] *= self.dt
	# prepare list of functions
	# 动态绑定奖励函数及其名称
	self.reward_functions = []
	self.reward_names = []
	# 📌reward_scales由legged_robot_config.py中的reward.scales类转化为字典得到
	for name, scale in self.reward_scales.items():
		# 单独处理终止惩罚
		if name=="termination":
			continue
		self.reward_names.append(name)
		name = '_reward_' + name
		# (2)
		self.reward_functions.append(getattr(self, name))

	# reward episode sums
	# 初始化累积奖励，存储每个环境的每个奖励项并初始化为零
	self.episode_sums = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
						 for name in self.reward_scales.keys()}
```

- `(1)`仿真的频率是可以调节的，而为了保证奖励在物理意义上的统一性，累积奖励的计算应该以单位时间而非仿真步数来衡量。因此在预处理阶段，所有奖励项的缩放因子都会按仿真时间步长折算。
- `(2)` `getattr(object, name)`函数直接通过字符串名称`name`获取对象`object`的属性或方法。

`compute_reward()`是奖励计算的核心执行函数：

```python
# 🔍 reward/"reward_functions"
# 📃 legged_robot.py
# [163]
def compute_reward(self):
    """ Compute rewards
        Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
        adds each terms to the episode sums and to the total reward
    """
    # 初始化奖励缓冲区
    self.rew_buf[:] = 0.
    # 遍历所有常规奖励项
    for i in range(len(self.reward_functions)):
    	# 计算奖励项的值
        name = self.reward_names[i]
        rew = self.reward_functions[i]() * self.reward_scales[name]
        # 累加到奖励缓冲区
        self.rew_buf += rew
        # (1) 累加到回合累积奖励
        self.episode_sums[name] += rew
    # (2) 可选奖励截断
    if self.cfg.rewards.only_positive_rewards:
        self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
    # add termination reward after clipping
    # 终止惩罚不受奖励截断影响，单独计算
    if "termination" in self.reward_scales:
        rew = self._reward_termination() * self.reward_scales["termination"]
        self.rew_buf += rew
        self.episode_sums["termination"] += rew
```

- `(1)`：`rew_buf[]`的计算周期是每个时间步，用于神经网络的更新；而`episode_sums[]`的计算周期是每个回合，其统计信息将用于日志记录（TensorBoard）。各个奖励项在回合尺度上的大小与占比可以为我们诊断训练问题、平衡奖励缩放因子提供依据。
- `(2)`：奖励截断将所有负奖励截断为零，是保障训练稳定性和探索效率的工程技巧。训练初期，策略完全随机，会产生大量负奖励且方差较大，奖励截断的作用在于：
  - 避免策略网络早期训练崩溃。过早地让策略接触大量负策略梯度容易扰乱学习方向，且抑制探索。
  - 简化价值网络需要学习的分布范围，稳定训练，促进收敛。

## 奖励缩放因子的调整策略

_待补充_

# 观测空间

足式机器人的观测空间输入主要来自以下几个部分：

- **本体感知**：机器人的内部状态，如关节运动信息、基座运动信息等。
- **外部感知**：环境信息，通常与机器人配备的传感器绑定，对复杂地形至关重要。例如由深度相机采集的深度图或处理得到的点云，也可与深度学习结合利用神经网络提取特征后输入。
- **任务相关输入**：用户指令，如期望的前进速度、转向角速度等。

`compute_observations()`函数负责构建机器人当前时刻的观测状态向量作为神经网络的输入：

```python
# 🔍 observation
# 📃 legged_robot.py
# [182]
def compute_observations(self):
	""" Computes observations
	"""
	# 所有环境的观测空间输入拼接而成的二维张量[num_envs, obs_dim]
	self.obs_buf = torch.cat((  self.base_lin_vel * self.obs_scales.lin_vel, # (1) 线速度(x, y, z)
								self.base_ang_vel  * self.obs_scales.ang_vel, # 角速度(roll, pitch, yaw)
								self.projected_gravity, # 重力向量在基座坐标系下的投影(x_b, y_b, z_b)
								self.commands[:, :3] * self.commands_scale, # 控制指令(前进速度, 侧移速度, 转向角速度)
								(self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos, # 相对于默认姿态的关节位置偏差[机器人关节数量]
								self.dof_vel * self.obs_scales.dof_vel, # 关节速度[机器人关节数量]
								self.actions # 上一步的动作[动作空间维度]
								),dim=-1)
	# add perceptive inputs if not blind
	# add noise if needed
	# (2) 添加噪声
	if self.add_noise:
		self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec
```

- `(1)`：观测空间的不同物理量数值范围存在差异，通过`obs_scales`类对其进行缩放到相近的范围（归一化），便于神经网络学习。
  - 重力向量仅用于观测姿态，本身是单位向量，无需归一化。
- `(2)`：噪声注入是在训练时对所有观测量添加`[-1, 1)`均匀分布的噪声，以提高策略对传感器噪声的稳定性的方法，测试时关闭。
  - `torch.rand_like()`生成一个与 `self.obs_buf` 形状完全相同的张量，每个元素是在 `[0, 1)` 区间内均匀分布的随机数。

观测缩放因子配置：

```python
# 🔍 "obs_scales"
# 📃 legged_robot_config.py
# [128]
class obs_scales:
	lin_vel = 2.0
	ang_vel = 0.25
	dof_pos = 1.0
	dof_vel = 0.05
	# 🐞 找不到这个参数的具体用法
	height_measurements = 5.0
```

## 观测噪声

`noise_scale_vec[]`由`_get_noise_scale_vec()`函数构建：

```python
# 🔍 "noise_scale_vec"
# 📃 legged_robot.py
# [182]
def _get_noise_scale_vec(self, cfg):
	""" Sets a vector used to scale the noise added to the observations.
		[NOTE]: Must be adapted when changing the observations structure

	Args:
		cfg (Dict): Environment config file

	Returns:
		[torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
	"""
	# 零初始化一个长度等于观测维度的噪声向量
	noise_vec = torch.zeros_like(self.obs_buf[0])
	# 获取噪声相关配置: 是否添加噪声, 基础噪声比例, 全局噪声强度
	self.add_noise = self.cfg.noise.add_noise
	noise_scales = self.cfg.noise.noise_scales
	noise_level = self.cfg.noise.noise_level
	# (1) 计算各观测噪声幅值: 基础噪声比例 * 全局噪声强度 * 观测缩放因子
	# 指令和动作是精确的，无需添加训练噪声
	noise_vec[:3] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel
	noise_vec[3:6] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
	noise_vec[6:9] = noise_scales.gravity * noise_level
	noise_vec[9:12] = 0. # commands
	noise_vec[12:12+self.num_actions] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
	noise_vec[12+self.num_actions:12+2*self.num_actions] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
	noise_vec[12+2*self.num_actions:12+3*self.num_actions] = 0. # previous actions

	return noise_vec
```

- `(1)`：观测噪声幅值计算仍需乘以观测缩放因子，因为基础噪声比例是基于原始物理量而非观测空间给出的，这种解耦让参数更具可解释性，便于人为赋值。

观测噪声配置：

```python
# 🔍 "noise"
# 📃 legged_robot_config.py
# [137]
class noise:
	add_noise = True
	noise_level = 1.0 # scales other values
	class noise_scales:
		dof_pos = 0.01
		dof_vel = 1.5
		lin_vel = 0.1
		ang_vel = 0.2
		gravity = 0.05
		# 🐞 找不到这个参数的具体用法
		height_measurements = 0.1
```

_未完待续_

# 动作空间

# 训练过程

# PPO超参数

```python
# [186]
class algorithm:
	# training params
	value_loss_coef = 1.0
	use_clipped_value_loss = True
	clip_param = 0.2
	entropy_coef = 0.012
	num_learning_epochs = 5
	num_mini_batches = 8 # mini batch size = num_envs*nsteps / nminibatches
	learning_rate = 1.e-3 #5.e-4
	schedule = 'adaptive' # could be adaptive, fixed
	gamma = 0.99
	lam = 0.95
	desired_kl = 0.01
	max_grad_norm = 1.
```

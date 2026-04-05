---
title: 【Legged Gym】四足机器人强化学习项目入门（更新中）
date: 2026-04-05
lastMod: 2026-04-05
summary: Legged Gym 是苏黎世联邦理工大学‌（ETH Zurich），主要用于训练四足机器人的运动控制策略 。‌‌
tags: [Legged Gym, IsaacGym, 四足机器人, PPO]
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

创建并激活虚拟环境`legged_gym`：

```bash
conda create -n legged-gym python=3.8
conda activate legged-gym
```

后续终端操作均在`legged-gym`虚拟环境下进行。

## 安装依赖

**安装 Pytorch**

Pytorch 是一个神经网络计算框架，用于模型训练和推理。下面是官方提供的安装版本：

```bash
pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

但是博主在解决问题的过程中调整为了如下版本：

```bash
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu121
```

如果遇到网络问题可设置清华镜像源：

```bash
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
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
# 切换版本分支以适配legged_gym
git checkout v1.0.2
pip install -e .
```

**安装 legged_gym**

`legged_gym`是项目本体。

```bash
git clone https://github.com/leggedrobotics/legged_gym.git
cd legged_gym
pip install -e .
```

由于项目版本较老，博主又遇到了一系列兼容问题，完整的适配方案如下：

```bash
pip install matplotlib==3.3.4
pip install numpy==1.19.5
pip install tensorboard
pip install 'setuptools<59.0.0'
pip install protobuf==3.20.3
```

## Auto DL

Auto DL是一个云算力租赁平台，若手头显卡性能较差可以考虑租赁。

基础使用教程推荐看B站《深度学习炼丹必修！AutoDL 租 GPU 全流程教学，跑通深度学习项目》

::bilibili{ # BV1s2YdzNESZ}

Auto DL 的环境配置方法基本不变，本节以博主自身实践为例介绍在Auto DL上需要注意的几点。

博主的租赁配置：

- RTX4090 (24GB)×1
- 基础镜像：PyTorch 2.3.0 & Python 3.12 (Ubuntu 22.04) & CUDA 12.1

环境配置时间较长，建议使用无卡模式开机。

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
# 克隆完成后建议改回true
```

解决一段时间后博主又遇到网络问题：

```text
fatal: unable to access 'https://github.com/unitreerobotics/unitree_rl_gym.git/': Failed to connect to github.com port 443 after 129504 ms: Connection timed out
```

无奈只能本地在项目网站上直接下载 zip ，再拖到`/autodl-tmp`下，运行解压命令：

```bash
unzip 文件名.zip
```

或者尝试使用镜像源：

```bash
git clone https://githubfast.com/leggedrobotics/legged_gym.git
```

# 试运行

运行以下命令对`anymal-c`四足机器人进行平坦地形训练：

```bash
python legged_gym/scripts/train.py --task=anymal_c_flat
```

但是训练时渲染的图形界面会崩溃报错：

```bash
段错误 (核心已转储)
```

疑似是内存分配的问题，必须使用`--headless`参数禁用图形渲染才能进行，这也能大大提高训练效率：

```bash
python legged_gym/scripts/train.py --task=anymal_c_flat --headless
```

Auto DL 训练时记得使用有卡模式开机，同时由于系统无图形化界面，也需要禁用图形渲染。

如果要开启图形渲染，需要根据显卡性能减少并行训练的环境数量：

```python
# 📃 /legged_gym/envs/legged_robot_config.py
# [3]
class LeggedRobotCfg(BaseConfig):
	class env:
		# 训练环境数量由4096改成64
		num_envs = 64
		# ...
```

也可以在命令中使用`--num_envs`临时配置训练环境数量：

```bash
python legged_gym/scripts/train.py --task=anymal_c_flat --num_envs=64 --headless
```

🐞 但是 Isaac Gym 窗口最大化必然会导致崩溃，暂无解决方案。

运行成功后，如果没有禁用图形渲染，将跳出图形界面显示训练过程（这里用`unitree RL Gym`的示意一下）：

![|500](https://inkem-1306784622.cos.accelerate.myqcloud.com/blog/pic/Pasted%20image%2020260327204950.png)

运行训练的终端将对每个回合更新显示训练数据，以下是博主第一轮训练倒数第二个回合的训练数据：

```text
# 回合迭代进度
Learning iteration 299/300
# 仿真速度(环境交互收集数据耗时, 网络梯度更新耗时)
Computation: 47909 steps/s (collection: 1.805s, learning 0.246s)
# 价值函数损失，更新Critic网络
Value function loss: 0.0025
# 替代损失，更新Actor网络
Surrogate loss: -0.0055
# 动作噪声标准差
Mean action noise std: 0.45
# 奖励函数部分
Mean reward: 17.74
Mean episode length: 988.48
Mean episode rew_action_rate: -0.0854
Mean episode rew_ang_vel_xy: -0.0462
Mean episode rew_collision: -0.0054
Mean episode rew_dof_acc: -0.0447
Mean episode rew_feet_air_time: -0.0970
Mean episode rew_lin_vel_z: -0.0262
Mean episode rew_orientation: -0.0166
Mean episode rew_torques: -0.1477
Mean episode rew_tracking_ang_vel: 0.4350
Mean episode rew_tracking_lin_vel: 0.8882
--------------------------------------------------------------------------------
# 总时间步
Total timesteps: 29491200
# 上一回合长度
Iteration time: 2.05s
# 已训练时间
Total time: 627.46s
# 预计剩余训练时间
ETA: 1.7s
```

奖励函数部分将在后续章节中详解，先直接运行`play`命令看看训练效果：

```bash
python legged_gym/scripts/play.py --task=anymal_c_flat
```

在 4096 个并行环境下，机器人训练效果显著：

<video controls width="800" preload="metadata">
  <source src="https://inkem-1306784622.cos.accelerate.myqcloud.com/blog/pic/output.mp4" type="video/mp4">
  您的浏览器不支持视频播放。
</video>

Auto DL 无图形化界面，训练效果的查看方法详见**训练操作-训练效果**一节。

# 训练操作

接下来我们引入更多的训练参数，实现更灵活的训练过程。

官方文档给出的参数说明如下：

- `--task`：训练任务；
- `--headless`：不渲染图形界面；
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

博主将结合具体使用场景对其进行讲解。

## 训练的保存与加载

每次运行训练时，默认情况下，训练过程都会保存在`logs/<experiment_name>/<date_time>_<run_name>/model_<checkpoint>.pt`，其中

- `experiment_name`：实验名称，不同的`task`有默认的实验名称，通过`--experiment_name`可以自定义实验名称。
  - 用于定义实验主题，例如区分不同的训练任务（平地行走/崎岖地形/爬楼梯）。
- `date_time`：训练开始的时间，例如3月29日16时06分46秒为`Mar29_16-06-46`。
- `run_name`：运行名称，默认为空，通过`--run_name`可以给定运行名称。
  - 在每个实验主题下，区分不同超参数或配置的训练尝试。
- `checkpoint`：检查点，训练过程每 50 回合保存一次模型参数。

启动一次新训练时，我们可以通过`--experiment_name`和`--run_name`区分这次训练：

```bash
python legged_gym/scripts/train.py --task=anymal_c_flat --experiment_name=exp1 --run_name=run1
```

从某一次训练中的模型参数加载时：

- `--resume`：声明训练从检查点加载。
- `--experiment_name`：指定训练所在的实验文件夹。
- `--load_run`：指定具体的训练，需要给出完整的`<date_time>_<run_name>`。
- `--checkpoint`：指定模型检查点。
- `--run_name`：不会继承，需要重新命名。

```bash
python legged_gym/scripts/train.py --task=anymal_c_flat --resume --experiment_name=exp1 --load_run=Mar29_16-46-08_run1 --checkpoint=500 --run_name=run2
```

不指定`--load_run`和`--checkpoint`则默认加载最新的一次训练及训练中最新的模型。

> 🐞
>
> - 如果所加载的训练没有迭代到最大回合数就停止（例如`Ctrl+C`中断），则无论是否保存过模型参数，加载后的训练对已经迭代的回合数不会保留。
> - 在已经迭代的回合数被正常保留的情况下，终端显示的预计剩余时间`ETA`会变为负数。

## 训练的配置

部分配置可以通过命令行参数临时调整：

- `--num_envs`：并行训练的环境个数。
- `--seed`：训练使用的随机数种子。
- `--max_iteration`：单次训练的最大迭代回合数。

其他配置，例如奖励缩放因子和PPO超参数等，则需在代码中修改后再启动训练。

## 训练的设备

- `--sim_device`：仿真计算设备，默认为GPU，指定 CPU 为 `--sim_device=cpu`。
- `--rl_device`：强化学习计算设备，默认为GPU，指定 CPU 为 `--rl_device=cpu`。

## 训练效果

运行`play.py`可查看训练效果：

```bash
python legged_gym/scripts/play.py --task=go2
```

指定模型的参数包括`--experiment` `--load_run` `--checkpoint`，用法与运行训练的命令相同。

由此，Auto DL 的模型则可以通过下载到本地，并存放于与 Auto DL 中一致的路径下后，在本地运行训练效果。

## 训练可视化

TensorBoard 是 TensorFlow 提供的可视化工具包，用于机器学习实验。

于仓库文件夹`\legged_gym`下打开终端，运行如下命令可以使用 TensorBoard 工具将日志文件夹中的训练过程可视化：

```bash
tensorboard --logdir=logs
```

进入终端输出的网址：

```text
TensorBoard 2.20.0 at http://localhost:6006/ (Press CTRL+C to quit)
```

![](https://inkem-1306784622.cos.accelerate.myqcloud.com/blog/pic/Pasted%20image%2020260329171203.png)

左侧边栏勾选要可视化的训练，中间的分类展示了不同数据指标：

- **Episode**：机器人的具体表现，包括各个奖励项。
- **Loss**：神经网络的学习过程，包括学习率和损失函数。
- **Perf**：训练过程的运行效率，包括数据收集和网络更新的占用时间和环境模拟帧数。
- **Policy**：策略输出指标，包括输出动作的标准差。
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
# [130]
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
# [815]
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

⚠️ 可以发现，`legged_robot.py`定义了强化学习环境具体的运行逻辑，而`legged_robot_config.py`是调整`legged_robot.py`运行参数的配置接口，是所有机器人训练的通用模板。`\envs`目录下分目录存储了所有机器人型号各自的`env`和`config`文件，它们继承自`legged_robot`文件，为训练专门的机器人型号做了特定的适配。`go2`的环境直接使用`legged_robot.py`，无需单独的`env`文件。现阶段后续代码的定位范围均在这两个文件下：

![](Pasted%20image%2020260403163543.png)

## 奖励函数

奖励在数学上默认包含正负两种情况，而奖励项的作用具体是激励还是惩罚在代码中由`scales`类处理，函数内部一概返回正奖励值。为了便于理解，对于给予负缩放因子的奖励项，博主将直接称呼其为惩罚以做区分。

### 任务目标奖励

```python
# [872]
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
# [877]
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
# [816]
def _reward_lin_vel_z(self):
    # Penalize z axis base linear velocity
    return torch.square(self.base_lin_vel[:, 2])
```

**线速度惩罚**$-\mathbf v_{b,z}^2$惩罚机器人基座在$Z$轴方向的线速度，保持基座高度稳定，抑制不必要的跳跃或下沉。

```python
# [820]
def _reward_ang_vel_xy(self):
    # Penalize xy axes base angular velocity
    return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
```

**角速度惩罚**$-||\boldsymbol\omega_{b,xy}||^2$惩罚机器人基座绕$X$轴和$Y$轴的角速度，保持基座姿态稳定，抑制侧翻与前后倾。

```python
# [824]
def _reward_orientation(self):
    # Penalize non flat base orientation
    return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)
```

**姿态惩罚**$-\|\mathbf g_{\mathrm{project}}\|^2$惩罚机器人重力向量在基座坐标系$XY$维度下的投影。相比角速度惩罚，姿态惩罚直接惩罚基座姿态倾斜的角度。

_配图_

同时惩罚角速度和角度是控制理论中 PD 控制在奖励函数中的体现，能够让机器人快速回正且抑制震荡。

```python
# [828]
def _reward_base_height(self):
    # Penalize base height away from target
    base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
    return torch.square(base_height - self.cfg.rewards.base_height_target)
```

**高度惩罚**$-(h_b-h^*_b)^2$惩罚机器人基座高度偏离目标值，期望基座保持在理想的工作高度，因具体任务和训练效果选择性设置。

`root_states[:, 2]`是机器人基座的$Z$坐标，`measured_heights[num_envs, num_samples]`是机器人测量的数个采样点的地形高度，`torch.mean()`取机器人基座距离所有采样点的相对高度的平均值作为基座高度。

```python
# [833]
def _reward_torques(self):
    # Penalize torques
    return torch.sum(torch.square(self.torques), dim=1)
```

**力矩惩罚**$-\|\boldsymbol\tau\|^2$惩罚关节力矩过大，鼓励节能和低功耗运动。

```python
# [837]
def _reward_dof_vel(self):
    # Penalize dof velocities
    return torch.sum(torch.square(self.dof_vel), dim=1)
```

**速度惩罚**$-\|\dot{\mathbf p}\|^2$惩罚关节速度过大，鼓励平缓运动。

```python
# [841]
def _reward_dof_acc(self):
    # Penalize dof accelerations
    return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)
```

**加速度惩罚**$-\|(\dot{\mathbf p}_{t-\Delta t}-\dot{\mathbf p}_t)/\Delta t\|^2$惩罚关节加速度（通过相邻仿真时间步的关节速度差计算）过大，进一步鼓励平缓运动，减少冲击。

力矩惩罚和加速度惩罚具有相关性，但力矩惩罚还考虑了静载荷（重力）与摩擦等因素，更注重减少过载与能耗。

```python
# [845]
def _reward_action_rate(self):
    # Penalize changes in actions
    return torch.sum(torch.square(self.last_actions - self.actions), dim=1)
```

**动作变化率惩罚**$-\|\mathbf a_{t-1}-\mathbf a_t\|^2$从策略网络层面，惩罚其相邻决策时间步输出的动作空间指令。

动作空间往往是关节位置空间，因此动作变化率惩罚和速度、加速度惩罚也具有相关性。但出于计算性能的限制，策略网络的决策时间步通常大于执行器的执行时间步（仿真时间步），故执行器具体的执行效果在策略网络的观测中存在盲区，加速度惩罚则从更细的粒度上弥补了这一点。

除此之外，动作变化率惩罚通过鼓励策略网络学到平滑的策略，避免过大的梯度，对训练稳定性也起到重要作用。

### 物理安全与限制惩罚

```python
# [849]
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
# [853]
def _reward_termination(self):
    # Terminal reward / penalty
    return self.reset_buf * ~self.time_out_buf
```

**终止惩罚**惩罚环境触发提前终止的次数，具体机制与终止条件的判断函数`check_termination()`有关：

```python
# 🔍 "reset_buf"/"time_out_buf"
# 📃 legged_robot.py
# [138]
def check_termination(self):
    """ Check if environments need to be reset
    """
    # 严重碰撞终止（接触力 > 1）
    self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)
    # 超时终止（到达回合时间限制）
    self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
    self.reset_buf |= self.time_out_buf
```

对每个环境，布尔张量`self.reset_buf`判断其是否触发任意终止条件，`time_out_buf`判断其是否触发超时终止条件（到达回合时间限制）。

_配图_

```python
# [857]
def _reward_dof_pos_limits(self):
    # Penalize dof positions too close to the limit
    out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.) # lower limit
    out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
    return torch.sum(out_of_limits, dim=1)
```

**关节限位惩罚**惩罚关节位置超出软限位的大小。

软限位是软件设置的限位，比物理硬件限位更严格，以留出惩罚作用幅度的余量。

```python
# [863]
def _reward_dof_vel_limits(self):
    # Penalize dof velocities too close to the limit
    # clip to max error = 1 rad/s per joint to avoid huge penalties
    return torch.sum((torch.abs(self.dof_vel) - self.dof_vel_limits*self.cfg.rewards.soft_dof_vel_limit).clip(min=0., max=1.), dim=1)
```

**关节速度限制惩罚**惩罚关节速度超出软限位的大小，但单个关节的最大惩罚截断为1。

因为位置（以及后文的力矩）在仿真器和硬件处理中都自带截断逻辑，但速度理论上可以无限增加（仿真器中），故需要在奖励计算时截断。过大的速度会导致仿真策略应用到实物因为摩擦力和空气阻力等而失效，同时避免速度惩罚过大，淹没其他奖励信号，甚至产生数值爆炸。

```python
# [868]
def _reward_torque_limits(self):
    # penalize torques too close to the limit
    return torch.sum((torch.abs(self.torques) - self.torque_limits*self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)
```

**关节力矩限制惩罚**惩罚关节力矩超出软限位的大小。

在速度惩罚和力矩惩罚的基础上，速度限制惩罚和力矩限制惩罚的作用在于进一步平衡奖惩机制，防止策略为了更高的任务奖励而牺牲安全性，对于超出安全边界的情况加大惩罚力度。

### 步态与行为风格

```python
# [882]
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
# [895]
def _reward_stumble(self):
    # Penalize feet hitting vertical surfaces
    return torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) >\
        5 *torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)
```

**绊倒惩罚**惩罚足端与垂直表面的接触，以足端受到的侧向力为判断依据，当其超过垂向力的5倍时，我们可以认为这是异常撞击而非正常推进（摩擦力）。

```python
# [900]
def _reward_stand_still(self):
    # Penalize motion at zero commands
    return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1) * (torch.norm(self.commands[:, :2], dim=1) < 0.1)
```

**站立姿态惩罚**是一个仅在命令机器人静止时（指令线速度小于0.1）被计算的条件奖励，鼓励机器人保持站立姿态。`default_dof_pos`是预设站立姿态下各关节位置，惩罚由当前各关节位置与预设位置之间的误差和计算而来。

```python
# [904]
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
# [544]
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

- `(1)`：仿真的频率是可以调节的，而为了保证奖励在物理意义上的统一性，累积奖励的计算应该以单位时间而非仿真步数来衡量。因此在预处理阶段，所有奖励项的缩放因子都会按仿真时间步长折算。
- `(2)`：`getattr(object, name)`函数直接通过字符串名称`name`获取对象`object`的属性或方法。

`compute_reward()`是奖励计算的核心执行函数：

```python
# 🔍 reward/"reward_functions"
# 📃 legged_robot.py
# [190]
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

## 奖励参数配置

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
	# 本体感知输入
	self.obs_buf = torch.cat((  self.base_lin_vel * self.obs_scales.lin_vel, # (1) 线速度(x, y, z)
								self.base_ang_vel  * self.obs_scales.ang_vel, # 角速度(roll, pitch, yaw)
								self.projected_gravity, # 重力向量在基座坐标系下的投影(x_b, y_b, z_b)
								self.commands[:, :3] * self.commands_scale, # 控制指令(前进速度, 侧移速度, 转向角速度)
								(self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos, # 相对于默认姿态的关节位置偏差[机器人关节数量]
								self.dof_vel * self.obs_scales.dof_vel, # 关节速度[机器人关节数量]
								self.actions # 上一步的动作[动作空间维度]
								),dim=-1)
	# add perceptive inputs if not blind
	# 额外拼接外部感知输入
	# 可选观测高度
	if self.cfg.terrain.measure_heights:
		# (2)
		heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements
		self.obs_buf = torch.cat((self.obs_buf, heights), dim=-1)
	# add noise if needed
	# (3) 添加噪声
	if self.add_noise:
		self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec
```

- `(1)`：观测空间的不同物理量数值范围存在差异，通过`obs_scales`类对其进行缩放到相近的范围（归一化），便于神经网络学习。
  - 重力向量仅用于观测姿态，本身是单位向量，无需归一化。
- `(2)`：观测的`heights[]`相比比奖励函数的`base_height[]`需要考虑传感器和基座之间的相对高度，故此处减0.5。
- `(3)`：噪声注入是在训练时对所有观测量添加`[-1, 1)`均匀分布的噪声，以提高策略对传感器噪声的稳定性的方法，测试时关闭。
  - `torch.rand_like()`生成一个与 `self.obs_buf` 形状完全相同的张量，每个元素是在 `[0, 1)` 区间内均匀分布的随机数。

观测缩放因子配置：

```python
# 🔍 "obs_scales"
# 📃 legged_robot_config.py
# [157]
class obs_scales:
	lin_vel = 2.0
	ang_vel = 0.25
	dof_pos = 1.0
	dof_vel = 0.05
	height_measurements = 5.0
```

## 观测噪声

`noise_scale_vec[]`由`_get_noise_scale_vec()`函数构建：

```python
# 🔍 "noise_scale_vec"
# 📃 legged_robot.py
# [455]
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
	# 指令和动作是精确的, 无需添加训练噪声
	noise_vec[:3] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel
	noise_vec[3:6] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
	noise_vec[6:9] = noise_scales.gravity * noise_level
	noise_vec[9:12] = 0. # commands
	noise_vec[12:24] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
	noise_vec[24:36] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
	noise_vec[36:48] = 0. # previous actions
	if self.cfg.terrain.measure_heights:
	    noise_vec[48:235] = noise_scales.height_measurements* noise_level * self.obs_scales.height_measurements
	return noise_vec
```

- `(1)`：观测噪声幅值计算仍需乘以观测缩放因子，因为基础噪声比例是基于原始物理量而非观测空间给出的，这种解耦让参数更具可解释性，便于人为赋值。

观测噪声参数配置：

```python
# 🔍 "cfg.noise"
# 📃 legged_robot_config.py
# [166]
class noise:
	add_noise = True
	noise_level = 1.0 # scales other values
	class noise_scales:
		dof_pos = 0.01
		dof_vel = 1.5
		lin_vel = 0.1
		ang_vel = 0.2
		gravity = 0.05
		height_measurements = 0.1
```

# 动作空间与控制器

足式机器人的动作空间根据表达的抽象程度可以划分为以下几个层级：

- **低层级**：关节扭矩
  - 无需底层控制器，动作直接发送给电机。
- **中层级**：关节速度/位置
  - 由一个运行频率更高的底层控制器来驱动关节电机，使其快速达到目标速度/位置。
- **高层级**：足端轨迹、步态参数
  - 通过轨迹规划将参数转换为轨迹，通过逆运动学将轨迹转换为关节位置，再由底层控制器执行。

可想而知，表达的抽象程度的提高将带来：

- 对机器人执行指令的能力要求提高，更加受限于机器人自身的运动模型。
- 机器人运动的灵活性降低。
- 动作空间维度和指令复杂度的降低让强化学习任务的训练变得更加简单。

`_compute_torques()`函数将动作空间的决策转换为关节扭矩：

```python
# 🔍 action, control
# 📃 legged_robot.py
# [353]
def _compute_torques(self, actions):
	""" Compute torques from actions.
		Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
		[NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

	Args:
		actions (torch.Tensor): Actions

	Returns:
		[torch.Tensor]: Torques sent to the simulation
	"""
	#pd controller
	# PD 控制器, 比例增益为p_gains, 微分增益为d_gains
	# (1) 动作缩放
	actions_scaled = actions * self.cfg.control.action_scale
	# 获取控制类型参数
	control_type = self.cfg.control.control_type
	# (2) P位置控制, V速度控制, T扭矩控制
	if control_type=="P":
		torques = self.p_gains*(actions_scaled + self.default_dof_pos - self.dof_pos) - self.d_gains*self.dof_vel
	elif control_type=="V":
		torques = self.p_gains*(actions_scaled - self.dof_vel) - self.d_gains*(self.dof_vel - self.last_dof_vel)/self.sim_params.dt
	elif control_type=="T":
		torques = actions_scaled
	else:
		raise NameError(f"Unknown controller type: {control_type}")
	# 扭矩裁剪, 使之不超过电机物理限制
	return torch.clip(torques, -self.torque_limits, self.torque_limits)
```

- `(1)`：策略网络的输出层通常使用输出范围有限的激活函数，例如`tanh`的范围是`(-1,1)`，需要缩放到实际物理范围。归一化的动作空间有助于同一网络架构泛化到不同型号的机器人。
- `(2)`：根据控制器提供的模式，策略网络可以使用关节位置、速度和力矩作为动作空间。
  - **位置控制**：动作被解释为默认关节位置的偏移量，比例项跟踪位置误差，微分项跟踪关节速度。
  - **速度控制**：动作被解释为关节速度，比例项跟踪速度误差，微分项跟踪关节加速度（替代为相邻时间步速度差除以时间步长）。
  - **扭矩控制**：动作被解释为关节扭矩，直接输出到电机。

控制器参数配置：

```python
# 🔍 "cfg.control"
# 📃 legged_robot_config.py
# [89]
class control:
	# 控制器类型
	control_type = 'P' # P: position, V: velocity, T: torques
	# PD Drive parameters:
	# 各关节刚度（比例增益）
	stiffness = {'joint_a': 10.0, 'joint_b': 15.}  # [N*m/rad]
	# 各关节阻尼（微分增益）
	damping = {'joint_a': 1.0, 'joint_b': 1.5}     # [N*m*s/rad]
	# action scale: target angle = actionScale * action + defaultAngle
	# 动作缩放因子
	action_scale = 0.5
	# decimation: Number of control action updates @ sim DT per policy DT
	# (1) 降采样因子
	decimation = 4
```

- `(1)`：降采样因子表示策略网络的决策时间步长对应几倍的仿真时间步长，用于调节策略网络的决策频率。策略频率越高，任务响应延迟越低，但训练难度和计算负载也随之上升。适当的策略频率设置也有助于训练过程的稳定性。

控制器参数处理：

```python
# 📃 legged_robot.py
def init_buffers(self):
	# ...
	# 🔍 "control"
	# [525]
	# joint positions offsets and PD gains
	# 初始化默认关节位置
	self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
	# 遍历所有关节
	for i in range(self.num_dofs):
		# 读取各关节名称及默认位置
		name = self.dof_names[i]
		angle = self.cfg.init_state.default_joint_angles[name]
		self.default_dof_pos[i] = angle
		found = False
		# 与刚度参数匹配并分配PD控制器增益
		for dof_name in self.cfg.control.stiffness.keys():
			# 模糊匹配
			if dof_name in name:
				self.p_gains[i] = self.cfg.control.stiffness[dof_name]
				self.d_gains[i] = self.cfg.control.damping[dof_name]
				found = True
		# 未匹配的关节设置为零增益
		if not found:
			self.p_gains[i] = 0.
			self.d_gains[i] = 0.
			if self.cfg.control.control_type in ["P", "V"]:
				print(f"PD gain of joint {name} were not defined, setting them to zero")
	self.default_dof_pos = self.default_dof_pos.unsqueeze(0)

# ...

# 🔍 "control"
# [728]
def _parse_cfg(self, cfg):
	# 决策时间步 = 降采样因子 * 仿真时间步
    self.dt = self.cfg.control.decimation * self.sim_params.dt
    # ...
```

# 仿真环境

## 环境创建

`create_sim()`是仿真环境创建的入口函数，负责建立完整的物理仿真世界：

```python
# [198]
def create_sim(self):
	""" Creates simulation, terrain and environments
	"""
	# 设置重力轴方向为Z轴
	self.up_axis_idx = 2 # 2 for z, 1 for y -> adapt gravity accordingly
	# (1) 创建仿真实例(计算设备, 渲染设备, 物理引擎, 仿真参数)
	self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
	# 创建地面平面
	self._create_ground_plane()
	# 创建并行环境
	self._create_envs()
```

- `(1)`：

仿真实例相关参数的传递链条较为复杂，难以通过简单的搜索和引用查找摸清。此处只给出参数的配置入口，具体传递将在**任务注册**一章中详解。

仿真实例参数配置：

```python
# 📃 legged_robot_config.py
# [154]
class sim:
	# 仿真时间步
	dt =  0.005
	substeps = 1
	gravity = [0., 0. ,-9.81]  # [m/s^2]
	up_axis = 1  # 0 is y, 1 is z

	class physx:
		num_threads = 10
		solver_type = 1  # 0: pgs, 1: tgs
		num_position_iterations = 4
		num_velocity_iterations = 0
		contact_offset = 0.01  # [m]
		rest_offset = 0.0   # [m]
		bounce_threshold_velocity = 0.5 #0.5 [m/s]
		max_depenetration_velocity = 1.0
		max_gpu_contact_pairs = 2**23 #2**24 -> needed for 8000 envs and more
		default_buffer_size_multiplier = 5
		contact_collection = 2 # 0: never, 1: last sub-step, 2: all sub-steps (default=2)
```

## 仿真过程

# 任务注册

# PPO

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

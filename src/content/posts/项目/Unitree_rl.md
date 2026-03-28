---
title: 【项目学习】Unitree RL GYM (Go2)（更新中）
date: 2026-03-28
lastMod: 2026-03-28
summary: 基于 Unitree 机器人实现强化学习的示例仓库，以四足机器人 Go2 为例
tags: [宇树, IsaacGym, 四足机器人, PPO]
category: 项目学习
---

项目地址：https://github.com/unitreerobotics/unitree_rl_gym

# 环境配置

参考文档：https://github.com/unitreerobotics/unitree_rl_gym/blob/main/doc/setup_zh.md

系统要求（博主配置）：

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

# 试运行

运行以下命令对`go2`四足机器人进行训练：

```bash
python legged_gym/scripts/train.py --task=go2
```

CUDA内存分配失败报错：

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

因为博主用的显卡太垃圾了，需要减少并行训练的环境数量，在`legged_robot_config.py`中修改：

_注释中将使用_`[*]`_对修改位置在代码中的原始行数进行必要的标记。_

```python
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

运行成功后将跳出图形界面显示训练过程：

![|500](https://inkem-1306784622.cos.accelerate.myqcloud.com/blog/pic/Pasted%20image%2020260327204950.png)

可以在命令中使用`--headless`禁用图形界面渲染，大大提高训练效率：

```bash
python legged_gym/scripts/train.py --task=go2 --headless
```

运行训练的终端将对每次迭代更新显示训练数据，以下是博主第一轮训练最后一次迭代前的训练数据：

```
迭代进度
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
足端腾空时间惩罚?
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
上一轮迭代时间
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

只看截图就知道效果奇差无比，机器人要么向后撑着，要么向前蛄蛹一下趴地上：

![|500](https://inkem-1306784622.cos.accelerate.myqcloud.com/blog/pic/Pasted%20image%2020260328194141.png)

影响训练效果的最直接的参数一方面为奖励项权重，另一方面为PPO算法的超参数，分别对应`legged_robot_config.py`中的`rewards`类和`algorithm`类：

```python
# [101]
class rewards:
	class scales:
		termination = -10.0
		tracking_lin_vel = 1.0
		tracking_ang_vel = 0.5
		lin_vel_z = -2.0
		ang_vel_xy = -0.05
		orientation = 1.0
		torques = -0.00001
		dof_vel = -0.
		dof_acc = -2.5e-7
		base_height = -1.0
		feet_air_time = 0.5
		collision = -0.5
		feet_stumble = -0.
		action_rate = -0.01
		stand_still = -0.
	only_positive_rewards = False # if true negative total rewards are clipped at zero (avoids early termination problems)
	tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)
	soft_dof_pos_limit = 1. # percentage of urdf limits, values above this limit are penalized
	soft_dof_vel_limit = 1.
	soft_torque_limit = 1.
	base_height_target = 0.25
	max_contact_force = 100.

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

接下来是博主漫长的调参数炼丹时间……

未完待续

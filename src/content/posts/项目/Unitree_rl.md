---
title: 【项目学习】Unitree RL GYM (Go2)（更新中）
date: 2026-03-28
lastMod: 2026-03-30
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

此时可以尝试减少并行训练的环境数量，在`legged_robot_config.py`中修改：

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

运行成功后，如果没有禁用图形渲染，将跳出图形界面显示训练过程：

![|500](https://inkem-1306784622.cos.accelerate.myqcloud.com/blog/pic/Pasted%20image%2020260327204950.png)

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

博主的 RTX 2050 只能跑 64 个并行环境，一轮训练几乎看不到效果，机器人要么向后撑着，要么向前蛄蛹一下趴地上：

![|500](https://inkem-1306784622.cos.accelerate.myqcloud.com/blog/pic/Pasted%20image%2020260328194141.png)

而 Auto DL 上可以跑满 4096 个并行环境，一轮训练效果明显，就是机器人是斜着走的：

<video controls width="800" preload="metadata">
  <source src="https://inkem-1306784622.cos.accelerate.myqcloud.com/blog/pic/output.mp4" type="video/mp4">
  您的浏览器不支持视频播放。
</video>

Auto DL 无图形化界面，训练效果的查看方法详见**训练-训练效果**一节。

# 训练

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
- `--max_iterations`：训练的最大迭代次数；
- `--sim_device`：仿真计算设备，指定 CPU 为 `--sim_device=cpu`；
- `--rl_device`：强化学习计算设备，指定 CPU 为 `--rl_device=cpu`。

## 训练的保存与加载

每次运行训练时，默认情况下，训练过程都会保存在`logs/<experiment_name>/<date_time>_<run_name>/model_<checkpoint>.pt`，其中

- `experiment_name`：实验名称，go2 机器人默认为`rough_go2`，通过`--experiment_name`可以自定义实验名称；
  - 用于定义实验主题，例如区分不同的训练任务（平地行走/崎岖地形/爬楼梯）。
- `date_time`：训练开始的时间，例如3月29日16时06分46秒为`Mar29_16-06-46`；
- `run_name`：运行名称，默认为空，通过`--run_name`可以给定运行名称；
  - 在每个实验主题下，区分不同超参数或配置的训练尝试。
- `checkpoint`：检查点，训练过程每 50 次迭代保存一次模型参数。

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

> [!bug]
>
> - 如果所加载的训练没有迭代到最大次数就停止（例如`Ctrl+C`中断），则无论是否保存过模型参数，加载后的训练迭代次数不会保留；
> - 在迭代次数被正常保留的情况下，终端显示的预计剩余时间`ETA`会变为负数。

## 训练的配置

部分配置可以通过命令行参数临时调整：

- `--num_envs`：并行训练的环境个数；
- `--seed`：训练使用的随机数种子；
- `--max_iteration`：单次训练的最大迭代次数。

其他配置，例如奖励权重和PPO超参数等，则需在代码中修改后再启动训练。

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

TensorBoard 是 TensorFlow 提供的可视化工具包，用于机器学习实验。使用如下命令安装：

```bash
conda install tensorboard
```

于仓库文件夹`unitree_rl_gym`下打开另一个终端，运行如下命令可以使用 TensorBoard 工具将日志文件夹中的训练过程可视化：

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

---
title: 【AdaptManip】自适应全身移动操作框架
date: 2026-06-26
summary: 无模仿学习、融合视觉+本体感知进行物体位姿估计的稳定搬运策略。
tags: [运动控制, 移动操作, 状态估计, 分层强化学习, 人形机器人, 论文精读]
category: 具身智能
---

- 论文标题：_AdaptManip: Learning Adaptive Whole-Body Object Lifting and Delivery with Online Recurrent State Estimation_
- 发表时间：2026.02.16
- 论文链接：https://arxiv.org/abs/2602.14363

> 博主正在火急火燎地摸索研究方向，因此会更新很杂的论文笔记。但就这段时间跟师兄做课题的经历而言，我可能会走多技能学习和移动操作方向，也差不多酝酿出了本科毕设课题的构想。上一篇MUJICA对应多技能，这一篇AdaptManip便是针对移动操作找的（特地先找了比较基础、算力需求小的）。

# 摘要

本文提出了自适应全身运动操作框架AdaptManip，该框架具备完全自主性，可应用于人形机器人执行集成“导航、物体抓取与递送”的任务。相较于以往依赖人类示范且对干扰敏感的基于模仿学习的方法，AdaptManip旨在通过强化学习，无需人类示范或远程操控数据，即可训练出稳健的运动操作策略。所提框架由三个紧密耦合的组件构成：（1）循环式物体状态估计器，能够在视野受限及存在遮挡的条件下，实时追踪被操作物体的状态；（2）全身基础运动策略，确保稳健的移动能力，并辅以残差操作控制，以实现稳定的物体抓取与递送；（3）基于激光雷达的机器人全局位置估计器，提供抗漂移的精准定位。所有组件均在仿真环境中通过强化学习进行训练，并以零样本迁移的方式部署至真实硬件平台。实验结果表明，在适应性和总体任务成功率方面，AdaptManip显著优于包括基于模仿学习的方法在内的各类基线方法，即使在存在遮挡的情况下，精确的物体状态估计仍能提升操作性能。此外，本研究还成功在人形机器人上实现了完全自主的真实世界导航、物体抓取与递送。

_主要关注「残差操作策略」奖励函数和「循环物体状态估计器」部分。_

![](https://inkem-1306784622.cos.accelerate.myqcloud.com/blog/pic/Pasted%20image%2020260628154211.png)

# 一、研究背景与任务描述

**移动操作**（Loco-Manipulation）的目标即让机器人实现移动与操作的协同，在执行行走、奔跑等移动任务的同时与物体进行复杂的物理交互，是让机器人从标准化工业场景走向居家服务等开放场景关键的一环。当前的主流思路之一是让机器人通过模仿人类动作获得基础的移动操作技能，但这类方法目前面临的关键瓶颈在于：

- 模仿学习严重依赖动作捕捉系统、遥操作和人体视频等提供的数据，这些数据通常包含全局状态和特权信息。
- 在上一点的限制下，策略往往遵循给定时间窗口的固定参考运动，因而难以适应物体滑移或掉落等自然失败情况。
- 假设物体始终可见，依赖纯视觉感知物体位姿，难以应对普遍存在的视觉遮挡情况。

而Adapt解决的核心问题便是，如何让双足人形机器人在完全没有外部动捕系统、依靠自身传感器的情况下，自主、稳健地完成“导航→抓取→搬运→放置”这一整套全身操作任务。任务分为三个阶段：

- 导航阶段：从初始位置导航到目标物体前方的指定位置，系统将利用机载激光雷达，通过基于激光雷达-惯性里程计的方法（FAST-LIO）来估计目标的相对位置。
- 抓取与抬升阶段：当机器人距离物体足够近时，开始抓取并抬起目标物体。机器人使用非关节式橡胶手依靠摩擦接触与物体交互。AdaptManip使用AprilTag实现对物体的视觉感知，该部分也可替换为任意基于视觉的6D位姿估计器。
- 搬运至目标阶段：将物体从初始位置搬运到目标位置，这一阶段物体经常被机器人自身遮挡，视觉信息变得不可靠。

导航和视觉感知部分不作为主要内容在本文介绍。

# 二、基于循环状态估计的移动操作学习

![](https://inkem-1306784622.cos.accelerate.myqcloud.com/blog/pic/Pasted%20image%2020260628153234.png)

AdaptManip使用状态估计与控制联合学习的方法，通过施加干扰让机器人在训练中经历各种失败，从而学会利用多模态感知（视觉 + 本体感知）持续推断物体状态并主动恢复。其中控制采用分层强化学习架构，下层基座行走策略$\pi_\text{wbc}$负责移动，上层残差操作策略$\pi_\text{res}$负责物体交互。

## 1. 基座行走策略

行走策略的Actor网络观测空间为

$$
\mathbf o^\text{actor}_t=\left[\boldsymbol\theta_j,\dot{\boldsymbol\theta}_j,\boldsymbol\omega,\mathbf g_\text{proj},\bar{\mathbf v},\bar\omega_z,\bar h,\mathbf a_{t-1}\right]
$$

| 符号                        | 含义                       |
| --------------------------- | -------------------------- |
| $\boldsymbol\theta_j$       | 关节位置                   |
| $\dot{\boldsymbol\theta}_j$ | 关节速度                   |
| $\boldsymbol\omega$         | 基座角速度                 |
| $\mathbf g_\text{proj}$     | 重力向量在机身坐标系的投影 |
| $\bar{\mathbf v}$           | 基座xy平面线速度指令       |
| $\bar\omega_z$              | 基座偏航角速度指令         |
| $\bar h$                    | 基座高度指令               |
| $\mathbf a_{t-1}$           | 上一步动作                 |

Critic网络使用包含真实基座线速度的特权观测，并且额外提供两步历史

$$
\mathbf o^\text{critic}_t=\left[\mathbf o^\text{actor}_{t-2:t},\mathbf v_{t-2:t}\right]
$$

动作空间$\mathbf a_t$为所有关节的位置偏移量，由PD控制器转换为力矩驱动电机。

奖励函数由指令追踪、步态塑形、运动正则化和约束违规惩罚四部分组成

$$
\begin{split}
r_\text{loco}&=\omega_\text{tr}\left(e^{-\|\mathbf v-\bar{\mathbf v}\|}+e^{-\|\omega_z-\bar\omega_z\|}+e^{-\|h-\bar h\|}\right)\\
&+\omega_\text{gait}\left(\sum_f\mathbb I_{v>0.1}(t_f-0.4)+e^{-0,5\sum_f|z_f-0,05|}\right)\\
&-\omega_\text{reg}\sum_j\left(\tau^2_j+\omega^2_j+\dot\omega^2_j+a^2_j+\dot a^2_j+\ddot a^2_j\right)\\
&-\omega_\text{vio}\left(\sum_f\mathbb I_c\|v_f\|+\sum_fv_ff_c+\|g_p\|^2+\omega_o+\tau_o\right)
\end{split}
$$

- 步态塑形：第一项奖励基座线速度大于0.1的足端腾空时间，$\mathbb I$的下标为计算该项的条件；第二项奖励足端离地高度。
- 约束违规惩罚：第一项惩罚足端触地时的滑动，第二项惩罚足端触地瞬间的冲击，第三项惩罚倾斜过度，后两项惩罚关节输出超限。

## 2. 残差操作策略

操作策略的Actor网络观测空间关注物体信息

$$
\mathbf o^\text{actor}_t=\left[\boldsymbol\theta_j,\dot{\boldsymbol\theta}_j,\boldsymbol\omega,\mathbf g_\text{proj},\tilde{\mathbf X}_\text{box}, \bar{\mathbf X}_\text{box},\mathbf a_{t-1}\right]
$$

$\tilde{\mathbf X}_\text{box}$和$\bar{\mathbf X}_\text{box}$分别为箱子6D位姿的估计和指令。

Critic网络同样使用特权观测（包括箱子的实际运动信息与手部和箱子的接触力）和两步历史

$$
\begin{gathered}
\mathbf o^\text{critic}_t=\left[\mathbf o^\text{actor}_{t-2:t},\mathbf o^\text{priv}_{t-2:t}\right]\\
\mathbf o^\text{priv}_t=\left[\mathbf X_\text{box},\mathbf v_\text{box},\boldsymbol\omega_\text{box},\mathbf f_\text{hand},\mathbf f_\text{box}\right]
\end{gathered}
$$

动作空间分为底层指令和上肢残差两部分。底层指令$\bar{\mathbf u}_\text{loco}=\left[\bar v_x,\bar v_y,\bar\omega_z,\bar h\right]$控制冻结的下层行走策略，上肢残差$\mathbf a_\text{upper}$在行走策略输出的目标基础上，对腰部和手臂关节输出残差。

操作策略的奖励函数在行走奖励$r_\text{loco}$的基础上新增了用于上肢操作的四个目标，包括运动学追踪、箱子稳定、接触力质量和防滑惩罚

$$
\begin{split}
r&=r_\text{loco}\\
&+\omega_\text{kin}\left(e^{-|\psi_\text{robot}-\psi_\text{box}|}+e^{-4\|p^\text{err}_\text{hand}\|}+e^{-1.5\|p^\text{err}_\text{root}\|}\right)\\
&+\omega_\text{box}\left(e^{-2\|p_\text{box}-p_\text{des}\|_1-\|q_\text{box}-q_\text{des}\|_1}+e^{-\|v_\text{root}-v_\text{box}\|_2}\right)\\
&+\omega_\text{con}\text{clamp}\left(\sum_h\|f_{\text{con},h}\|\mathbb I_\text{box},0,1\right)\\
&-\omega_\text{con}\sum_h\min(0,v_{\text{hand},z}-v_{\text{box},z})
\end{split}
$$

- 运动学追踪：奖励机器人和箱子朝向一致 + 手部位置接近期望抓取点 + 躯干位置接近期望位置。
- 箱子稳定：奖励箱子位姿接近期望位姿 + 躯干和箱子线速度同步。
- 接触力质量：奖励双手均与箱子有足够的接触力。
- 防滑惩罚：惩罚手部相对于箱子的切向运动。

接触力质量和防滑惩罚是让策略学会补救箱子松脱趋势的关键。

失败终止条件包括

- 超过20秒未完成（超时）。
- 机器人倾斜超过60°（摔倒）。
- 躯干高度低于0.15m（摔倒）。
- 箱子高度低于0.25m（箱子掉落）。

## 3. 循环物体状态估计器

仅靠视觉会由于操作过程中产生的相机遮挡，难以可靠地感知物体位姿，而强行要求物体一直可见又会导致机器人产生不自然的姿势，增加摔倒风险。AdaptManip提出了一种融合视觉观测与本体感知的在线物体状态估计方法，其受到人类搬运物体的策略启发：在抓取前确定一次物体的位姿，搬起后凭借手感即可确认物体还在不在手里。

状态估计器的网络架构为LSTM+MLP输出头，在每个时间步接收视觉测量、本体感知和历史动作三类信息。LSTM的内部记忆单元负责记住箱子最后已知的位置（来自RGB-D相机），在视觉缺失时依靠本体感知和动作历史来递推更新，MLP将LSTM的隐状态映射为最终的6D位姿估计，利用仿真器给出的位姿进行监督学习训练。

状态估计器与操作策略同步训练，但策略的物体位姿估计输入$\mathbf X_\text{in}$采用课程学习的方式

$$
\mathbf X_\text{in}=w\tilde{\mathbf X}_\text{box}+(1-w)\mathbf x_\text{box},\quad w=\min(t/T,1)
$$

权重因子$w$随训练迭代次数从0增大到1，即训练初期策略输入直接使用真实位姿，并逐步从真值切换为估计器输出的带噪声位姿。

但依赖本体感知估计物体位姿的局限在于，如果物体在手中发生相对滑动，估计器会逐渐漂移。一旦发现漂移，策略会选择松开重新抓取。引入触觉/力觉传感器可能改善这一问题。

## 4. 域随机化与抗干扰训练

引入抓取相关的域随机化因素是提升机器人抗干扰能力的关键，包括箱子的尺度、质量、质心位置、静/动摩擦因数、弹性系数等。除此之外，AdaptManip还会对策略输入的物体位姿采取加噪和随机掩码，模拟传感器误差与故障。

_论文没有具体说明域随机化的采样尺度（回合/时间步），但博主个人认为部分随机化因素在时间步尺度上采样更好，一方面人为制造抓取过程中的失败情况，另一方面搬运过程中箱子的质心、摩擦系数等确实存在变化的情况（不过可能无法用简单的随机化模拟，需要更复杂的设计）。_

# 三、总结

本文提出了一种新型框架AdaptManip，用于完成全身人形机器人的移动操作任务。它融合了激光雷达、视觉和本体感觉等多种感官信息，持续更新对箱子位姿的判断，并借助分层强化学习训练有效的策略，使机器人能够将箱子从起点拾起并搬运到目标位置。虽然该方法在箱子搬运任务上表现良好，但后续仍有不少值得探索的方向，例如尝试更复杂的任务流程、引入更多传感器来提升物体状态感知能力，或采用灵巧手来实现更精细的操作。

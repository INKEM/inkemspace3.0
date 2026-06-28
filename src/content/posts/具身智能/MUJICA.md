---
title: 【MUJICA】面向轮足机器人的统一多技能控制框架
date: 2026-06-26
summary: 引入状态估计器和精确电机约束训练盲控策略，技能选择器实现多技能自主切换。
tags:
  [决策规划, 状态估计, 安全强化学习, 分层强化学习, 非对称Actor-Critic, PPO, 轮足机器人, 论文精读]
category: 具身智能
---

::bilibili{#BV1Dm5Y6JEFV}

- 论文标题：_MUJICA: Multi-skill Unified Joint Integration of Control Architecture for Wheeled-Legged Robots_
- 发表时间：2026.05.13
- 论文链接：https://arxiv.org/pdf/2605.13058
- 补充知识：盲控策略 | 非对称Actor-Critc | 惩罚式PPO | SwAV对比学习

# 摘要

与足式机器人相比，轮腿式机器人在穿越复杂地形方面具有广阔的应用前景，并提供了优越的机动性能。然而，轮腿式机器人必须有效地平衡轮式驱动和腿式控制。此外，由于有噪声的本体感觉和现实世界中的电机限制，在电机的峰值性能下实现鲁棒和自适应的运动仍然具有挑战性。本文提出了多技能统一联合集成控制架构（MUJICA），这是一种统一的、完全本体感知的轮腿式机器人控制框架，它在单一策略内集成了各种低级技能，包括全方位移动、高平台攀爬和跌倒恢复。所有技能由独特的指标变量区分，并通过精确的直流电机约束建模进行联合训练。此外，学习一个高级技能选择器，根据本体感觉动态地选择最优技能，从而实现对周围环境的自适应反应。因此，MUJICA增强了Sim2Real的鲁棒性，并实现了不同运动模式之间的无缝转换，有利于对环境的自主调整。

![](https://inkem-1306784622.cos.accelerate.myqcloud.com/blog/pic/Pasted%20image%2020260626151000.png)

# 一、研究背景

轮足机器人结合了轮式的效率和足式的越障能力，在平坦地面能够快速移动，在崎岖地形能够跨越障碍。该能力在控制层面主要依赖于针对不同的地形选用不同的技能。但是现有的轮足式控制方法存在如下局限：

- 多技能盲控能力弱。**盲控策略**（Blind Policy）即仅依赖本体感知的控制策略，目前大多数盲控只能处理相似技能（平地走和斜坡走）或低难度技能。
- 仅关注技能多样性，但技能切换依赖人为手动切换或定义规则，缺乏自主适应性。
- 没有对直流电机的约束或过于简化，忽略直流电机的力矩输出能力随速度和位置变化这一物理事实。

针对上述局限，MUJICA分别提出以下解决方案：

- 通过状态估计器从本体感知中推断环境信息，增强盲控策略的感知能力。
- 引入可学习的技能选择器，实现自主技能切换。
- 建立精确的直流电机约束模型并纳入P3O训练框架，确保安全部署。

# 二、问题表述

![](https://inkem-1306784622.cos.accelerate.myqcloud.com/blog/pic/Pasted%20image%2020260626151106.png)

## 1. C-POMDP框架

轮足机器人在非结构化环境中作业时，由于传感器噪声和物理限制，无法获得完整的环境状态，同时必须满足安全性约束才能可靠部署到现实世界，因此使用**带约束的部分可观测马尔可夫决策过程**（Constrained Partially Observable MDP，C-POMDP）对问题进行建模

$$
\begin{gathered}
\max\mathbb E_\pi\left[\sum^\infty_{t=0}\gamma^tR(\mathbf s_t,\mathbf a_t,\mathbf s_{t+1})\right]\\
\text{s.t.}\quad\mathbb E_\pi\left[\sum^\infty_{t=0}\gamma^tC_i(\mathbf s_t,\mathbf a_t,\mathbf s_{t+1})\right]\leq\delta_i,\forall i\in\{1,\cdots,k\}
\end{gathered}
$$

- $R$：奖励函数。
- $C_i$：第$i$个约束成本函数，对应第$i$个安全限制。
- $\delta_i$：第$i$个约束的允许上限。
- $\gamma$：折扣因子。

## 2. 观测空间与状态空间

MUJICA使用了**非对称演员-评论家**（Asymmetric Actor-Critic）架构，旨在保证部署可行性的同时大幅提升训练效率。具体而言，Actor网络只能访问部署时可用的本体感知，而仅用于训练的Critic网络可以访问仿真环境能够获取的全状态信息。本文的观测空间特指Actor网络的观测空间，而状态空间则被定义为Critic网络的特权观测空间。

Actor网络的观测空间

$$
\mathbf o_t=\left[\boldsymbol\omega^b_t,\mathbf g^b_t,\mathbf{cmd}_t,\mathbf q_t,\dot{\mathbf q}_t,\mathbf a_{t-1},\zeta_t\right]^T\\
$$

| 符号                    | 含义                       |
| ----------------------- | -------------------------- |
| $\boldsymbol\omega^b_t$ | 基座角速度                 |
| $\boldsymbol g^b_t$     | 基座重力方向               |
| $\mathbf{cmd}_t$        | 指令（线速度+角速度）      |
| $\boldsymbol q_t$       | 关节角度                   |
| $\dot{\mathbf q}_t$     | 关节角速度                 |
| $\mathbf a_{t-1}$       | 上一步动作                 |
| $\zeta_t$               | 技能指示变量（详见第三章） |

_但为了让策略在盲控条件下依然能“感知”到看不见的环境信息，MUJICA在Actor网络的原始观测空间基础之上拼接了状态估计器的输出作为实际输入，详见第三章。_

Critic网络的状态空间

$$
\mathbf s_t\triangleq\mathbf o^{\text{priv}}_t=\left[\mathbf o_t,\mathbf v_t,\mathbf c_t,\mathbf u_t,\mathbf h_t\right]^T\\
$$

| 符号          | 含义         |
| ------------- | ------------ |
| $\mathbf v_t$ | 基座线速度   |
| $\mathbf c_t$ | 部件碰撞状态 |
| $\mathbf u_t$ | 轮地距离     |
| $\mathbf h_t$ | 局部高程图   |

## 3. 动作空间

动作空间定义为所有关节电机的控制命令：

- 腿部关节：相对于默认姿态的角度偏移量。
- 轮子关节：轮子转速。

动作由**PD控制器**（PD Controller）转换为实际力矩，腿部按位置控制，轮子按速度控制

$$
\tau^i_t=
\begin{cases}
K^i_d(q^i_t-a^i_t),&\text{if }i=\text{wheel}\\
K^i_p(q^i_t-a^i_t)-K^i_d\dot q^i_t,&\text{otherwise}
\end{cases}
$$

# 三、方法

## 1. 状态估计器

**状态估计器**（State Estimator）的作用是为盲控策略提供基于本体感知推断的环境信息，由门控循环单元（GRU）和全连接层（NN）组成。估计器接收NN对过去$H$步观测的特征提取结果和隐状态（上一步的估计量）作为输入，输出当前步的估计量

$$
\mathbf f_t=\text{GRU}(\text{NN}(\mathbf o_{t-H:t}),\mathbf f_{t-1})
$$

估计量替代了Critic网络的特权观测内容，作为Actor网络的补充观测

$$
\mathbf f_t=\left[\hat{\mathbf v}_t,\hat{\mathbf c}_t,\hat{\mathbf u}_t,\hat{\mathbf e}_t\right]^T
$$

| 符号                | 含义         |
| ------------------- | ------------ |
| $\hat{\mathbf v}_t$ | 基座线速度   |
| $\hat{\mathbf c}_t$ | 部件碰撞概率 |
| $\hat{\mathbf u}_t$ | 轮地距离     |
| $\hat{\mathbf e}_t$ | 隐式环境特征 |

其中$\hat{\mathbf v}_t$、$\hat{\mathbf c}_t$和$\hat{\mathbf u}_t$可以由仿真器给出明确的标签，使用监督学习训练

$$
\mathcal L^{\text{Pred}}=\mathcal L_{\text{MSE}}(\mathbf v_t,\hat{\mathbf v}_t)+\mathcal L_{\text{BCE}}(\mathbf c_t,\hat{\mathbf c}_t)+\mathcal L_{\text{MSE}}(\mathbf u_t,\hat{\mathbf u}_t)
$$

- MSE：均方误差。
- BCE：二元交叉熵。

而$\hat{\mathbf e}_t$并未采用特权观测中的高程图作为估计目标，因为仅凭本体观测根本无法唯一确定地形起伏，强行推断会导致估计误差极大，污染策略输入。因此MUJICA不要求估计器生成式重建完整的地形高程图，而是仅提取有助于决策的地形线索，学会区分不同地形类别的判别性特征。

$\hat{\mathbf e}_t$没有明确的真值标签，MUJICA采用**视图间任务交换**（Swapping Assignments between Views，SwAV）的对比学习方法进行训练。简而言之，SwAV会引入一个参考编码器，将下一步的观测$\mathbf o_{t+1}$映射为潜状态$\mathbf e_t$。而状态估计器则作为在线编码器，将过去$H$步的观测$\mathbf o_{t-H:t}$映射为$\hat{\mathbf e}_t$。SwAV损失则将二者的映射结果在特征空间中聚类对齐。

在线编码器和参考编码器的观测不对称性基于物理世界在时间上的连续平滑性而设计，让正样本带有时间关联性，迫使在线编码器提取与未来演化有关的环境特征，同时避免算法通过学到两个一模一样的编码器作弊。

最终状态估计器的总损失为监督学习和SwAV学习的损失之和

$$
\mathcal L^{\text{Estimate}}=\mathcal L^{\text{Pred}}+\mathcal L^{\text{SwAV}}(\mathbf e_t,\hat{\mathbf e}_t)
$$

状态估计器与Critic网络的特权观测都能帮助盲控策略“感知”到环境信息，关键区别在于状态估计器提供可部署的即时感知，而特权观测让Critic网络能够在训练中给予长期的价值引导。

## 2. 惩罚式近端策略优化

**惩罚式近端策略优化**（Penalized Proximal Policy Optimization，P3O）算法是标准PPO算法的改进，专门用于解决带有约束的优化问题。P3O的损失函数为（省略熵正则化与价值损失）

$$
\mathcal L^{\text{P3O}}(\theta)=\mathcal L^{\text{CLIP}}_R(\theta)+\kappa\sum^k_{i=1}\max\{0,\mathcal L^{\text{CLIP}}_{C_i}(\theta)\}
$$

其中$\mathcal L^{\text{CLIP}}_R(\theta)$是继承自标准PPO的奖励目标函数，但为了统一处理最大化奖励和最小化约束违规的目标，此处将奖励写为最小化问题下的负值形式

$$
\mathcal L^{\text{CLIP}}_R(\theta)=\mathbb E_{s\sim d^\pi,a\sim\pi}[-\min\{r(\theta)A^\pi_R(s,a),\text{clip}(r(\theta),1-\epsilon,1+\epsilon)A^\pi_R(s,a)\}]
$$

$\mathcal L^{\text{CLIP}}_{C_i}(\theta)$是P3O添加的约束目标函数

$$
\mathcal L^{\text{CLIP}}_{C_i}(\theta)=\mathbb E_{s\sim d^\pi,a\sim\pi}[\max\{r(\theta)A^\pi_{C_i}(s,a),\text{clip}(r(\theta),1-\epsilon,1+\epsilon)A^\pi_{C_i}(s,a)\}+(1-\gamma)(J_{C_i}(\pi)-\delta_i)]
$$

**约束优势**$A^\pi_{C_i}$衡量“该动作比平均动作多违反了多少约束”，用于惩罚当下的违规动作。奖励目标函数通过最小化操作保守更新防止策略过度优化奖励，而约束目标函数通过最大化操作激进惩罚防止策略过度违反约束。

$$
A^\pi_{C_i}=\mathbb E\left[\sum^\infty_{t=0}\gamma^tC_i(s_t,a_t,s_{t+1})\right]-V^\pi_{C_i}(s_t)
$$

**期望约束**$J_{C_i}(\pi)$衡量策略整体违反约束的程度，强制策略整体向安全区域偏移，保证约束优势的纠正有效。$\delta_i$是第$i$个约束的安全阈值，$(1-\gamma)$将折扣累积和归一化为平均约束成本。

$$
J_{C_i}(\pi)=\mathbb E_{s\sim d^\pi,a\sim\pi}\left[\sum^\infty_{t=0}\gamma^tC_i(s_t,a_t,s_{t+1})\right]
$$

**动态惩罚系数**$\kappa$根据策略的期望约束动态调整，权衡策略更新对安全和性能的重视程度，其更新规则如下

$$
\kappa\leftarrow\kappa\cdot\exp(\beta\cdot\text{mean}(J_{C_i}(\pi)-\delta_i))
$$

下图展示了无电机约束（上）和有电机约束（下）的力矩-速度分布。

![](https://inkem-1306784622.cos.accelerate.myqcloud.com/blog/pic/Pasted%20image%2020260626153251.png)

## 3. 多技能统一训练

为了在同一个策略网络中训练多种技能，MUJICA为每个技能按独热编码的方式分配了一个技能指示变量$\zeta_t$，该变量将作为观测$\mathbf o_t$的一部分输入给策略网络。通过这种方式，策略网络可以在不同技能之间共享底层运动知识，又能执行差异巨大的高层行为。

由于不同技能的目标不同，奖励函数和约束函数也在技能之间采取部分共享的差异化设计。对于全向移动（i）、爬高台（ii）和摔倒恢复（iii）三个技能，奖励与约束的设计如下表所示。

| 奖励或约束       | 公式                                                                                                                                                               | 适用技能 | 备注                                                                                                           |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------- | -------------------------------------------------------------------------------------------------------------- |
| 速度追踪奖励     | $\exp(-\|\text{cmd}_{xy,t}-\mathbf v_{xy,t}\|^2/\sigma^2)$                                                                                                         | i,ii     | 无。                                                                                                           |
| 角速度追踪奖励   | $\exp(-\|\text{cmd}_{\omega,t}-\omega_{z,t}\|^2/\sigma^2)$                                                                                                         | i        | 只有全向移动需要追踪角速度，爬高台过程不需要复杂的自旋。                                                       |
| 重力奖励         | $\exp(-\angle(\mathbf g^b_t,\mathbf g^\text{world})/\sigma^2)$                                                                                                     | iii      | 衡量机器人基座姿态是否摆正。                                                                                   |
| 关节位置奖励     | $\begin{gathered}\exp(-\|\mathbf q-\mathbf q_\text{stand}\|^2/\sigma^2)\\\text{if}\lvert\angle(\mathbf g^b_t,\mathbf g^\text{world})\rvert<\epsilon\end{gathered}$ | iii      | 在姿态摆正的基础之上要求腿部关节恢复到预定义的站立姿态。                                                       |
| 直流电机力矩约束 | $\sum^{16}_{i=1}\mathbf1_{\lvert\tau^i_t\rvert>\tau^i_\text{limit}}$                                                                                               | i,ii,iii | 统计当前时刻超出电机安全输出能力的关节数量。                                                                   |
| 碰撞约束         | $\sum_ic^i_t,i=\text{thigh,calf}$                                                                                                                                  | i,ii     | 惩罚大腿和小腿与环境发生碰撞，让机器人尽量用轮子接触地形。摔倒恢复过程的腿部碰撞无法避免，因此不需要碰撞约束。 |

针对直流电机力矩约束，MUJICA未简单采用统一的力矩限制，而是结合官方电机手册建立了精确的电机力矩-速度-位置关系模型，力矩限制将随电机运行状态而变化。

![](https://inkem-1306784622.cos.accelerate.myqcloud.com/blog/pic/Pasted%20image%2020260626154538.png)

## 4. 技能选择器

**技能选择器**（Skill Selector）替代人为手动切换或定义规则，让机器人自行选择当下应执行的技能。技能选择器的输入是过去$H$步的观测历史$\mathbf o_{t-H:t}$（不包含$\zeta_t$），输出技能的概率分布。

低层技能策略和高层技能选择器分为两个阶段单独训练。S1阶段训练技能策略直至收敛，S2阶段冻结技能策略，训练技能选择器。技能选择器的训练只采用线速度跟踪奖励，即选择技能的评价标准是机器人是否以指令速度稳定前进，这将迫使选择器学会在平地上全向移动、遇到高台时选择爬高台、摔倒时选择恢复。

# 四、总结

MUJICA的核心创新在于针对轮足机器人控制的三大挑战给出了系统性的解决方案：状态估计器让盲控策略在无外部感知的条件下仍能“感知”地形与碰撞，解决了多技能盲控能力弱的问题；可学习的技能选择器替代了人工规则，实现了自主技能切换；精确的DC电机约束+P3O训练框架确保了极限工况下的安全部署，最终实现了安全、高效、自主的轮足机器人控制。下一步工作是将该框架扩展到更广泛的运动技能，并在非结构化地形上实现自适应协调。

---
title: 【深度强化学习】#4 DPG：确定性策略梯度
date: 2025-10-17
summary: 以确定性策略为输出的Actor-Critic方法
tags: [Actor-Critic, 策略梯度, 离轨策略]
category: 深度强化学习
---

在此之前，我们介绍的策略梯度算法均基于随机策略。但是在连续动作空间问题中，以动作概率分布为输出的随机策略梯度具有一些核心问题：

- 动作采样和对数概率计算占用大量计算开销。
- 只要涉及对动作空间的采样，就无法根除方差问题。
- 策略的探索性由其方差控制，但策略网络会学会降低方差来减少风险。
- 对数操作使得概率较小的动作会获得较大的梯度，容易使网络参数剧烈变化。

**DPG**（Deterministic Policy Gradient，确定性策略梯度）系列算法以确定性策略$\mu(s)=a$为输出，将上述问题悉数解决，在确定性连续控制任务上超越了A2C等随机Actor Critic方法。

# DPG

在随机策略梯度中，目标函数需要对状态和动作的概率分布均进行积分，即

$$
J(\pi_\theta)=\int_\mathcal S\rho(s)\int_\mathcal A\pi_\theta(a|s)Q^{\pi_\theta}(s,a)\mathrm da\mathrm ds
$$

其中$\rho$为状态的概率分布，通常是初始状态概率分布$\rho_0$，因为我们更关心智能体从开始到结束的整个表现。

但是在确定性策略的情况下，目标函数得到了简化，$V^{\mu_\theta}(s)$与$Q^{\mu_\theta}(s,\mu_\theta(s))$等价，只需对状态空间进行积分

$$
\begin{split}
J(\mu_\theta)&=\int_\mathcal S\rho_0(s)Q^{\mu_\theta}(s,\mu_\theta(s))\mathrm ds\\
&=\mathbb E_{s\sim\rho_0}[Q^{\mu_\theta}(s,\mu_\theta(s))]
\end{split}
$$

## 确定性策略梯度定理

现在来证明确定性策略梯度定理。首先计算$Q^{\mu_\theta}(s,\mu_\theta(s))$关于$\theta$的梯度，$\theta$会通过两条路径影响到$Q^{\mu_\theta}(s,\mu_\theta(s))$的值：

- 在状态$s$下，$\mu_\theta$选择的动作$a$不同，对应的$Q$值不同，即$\theta\rightarrow\mu_\theta\rightarrow a\rightarrow Q$。
- 即使在状态$s$下$\mu_\theta$选择动作$a$不受影响，后续状态$\mu_\theta$的决策有所改变，对应的$Q$值也不同，即$\theta\rightarrow\mu_\theta\rightarrow Q$。

因而$Q^{\mu_\theta}(s,a)|_{a=\mu_\theta(s)}$的梯度也是改变$\mu_\theta$以改变$a$（第一项）和在固定$a$不变的条件下改变$\mu_\theta$（第二项）这两种情况所带来的梯度之和

$$
\nabla_\theta Q^{\mu_\theta}(s,\mu_\theta(s))=\nabla_\theta\mu_\theta(s)\nabla_aQ^{\mu_\theta}(s,a)|_{a=\mu_\theta(s)}+\nabla_\theta Q^{\mu_\theta}(s,a)|_{a=\mu_\theta(s)}
$$

数学记法表意有限，容易让人混淆，将其与语言描述对应上即可。

现在，我们有目标函数的梯度如下

$$
\begin{split}
\nabla_\theta J(\mu_\theta)&=\mathbb E_{s\sim\rho_0}[\nabla_\theta Q^{\mu_\theta}(s,\mu_\theta(s))]\\
&=\mathbb E_{s\sim\rho_0}[\underset{A(s)}{\underbrace{\nabla_\theta\mu_\theta(s)\nabla_aQ^{\mu_\theta}(s,a)|_{a=\mu_\theta(s)}}}]+\mathbb E_{s\sim\rho_0}[\underset{B(s)}{\underbrace{\nabla_\theta Q^{\mu_\theta}(s,a)|_{a=\mu_\theta(s)}}}]
\end{split}
$$

针对第二项，我们有

$$
\begin{split}
\mathbb E_{s\sim\rho_0}[B(s)]&=\mathbb E_{s\sim\rho_0}[\nabla_\theta Q^{\mu_\theta}(s,a)|_{a=\mu_\theta(s)}]\\
(s,a固定，梯度只作用于后继状态价值)&=\mathbb E_{s\sim\rho_0}[\gamma\mathbb E_{s'\sim p(\cdot|s,\mu_\theta(s))}\nabla_\theta V^{\mu_\theta}(s')]\\
(合并初始状态与后继状态的概率分布)&=\gamma\mathbb E_{s'\sim p(s_1=s'|\rho_0,\mu_\theta)}[\nabla_\theta V^{\mu_\theta}(s')]\\
(建立递归关系)&=\gamma(\mathbb E_{s\sim p(s_1=s|\rho_0,\mu_\theta)}[A(s)]+\underset{C_1}{\underbrace{\mathbb E_{s\sim p(s_1=s|\rho_0,\mu_\theta)}[B(s)]}})
\end{split}
$$

设$C_t=\mathbb E_{s\sim p(s_t=s|\rho_0,\mu_\theta)}[B(s)]$，则$C_0=\mathbb E_{s\sim\rho_0}[B(s)]$，进而有

$$
\begin{split}
C_0&=\gamma(\mathbb E_{s\sim p(s_1=s|\rho_0,\mu_\theta)}[A(s)]+C_1)\\
&=\gamma[\mathbb E_{s\sim p(s_1=s|\rho_0,\mu_\theta)}[A(s)]+\gamma(\mathbb E_{s\sim p(s_2=s|\rho_0,\mu_\theta)}[A(s)]+C_2)]\\
&=\gamma\mathbb E_{s\sim p(s_1=s|\rho_0,\mu_\theta)}[A(s)]+\gamma^2\mathbb E_{s\sim p(s_2=s|\rho_0,\mu_\theta)}[A(s)]+\gamma^2C_2\\
&=\sum^\infty_{t=1}\gamma^t\mathbb E_{s\sim p(s_t=s|\rho_0,\mu_\theta)}[A(s)]+\underset{=0}{\underbrace{\lim_{T\rightarrow\infty}\gamma^TC^T}}
\end{split}
$$

代回目标函数的梯度可得

$$
\begin{split}
\nabla_\theta J(\mu_\theta)&=\mathbb E_{s\sim\rho_0}[A(s)]+\sum^\infty_{t=1}\gamma^t\mathbb E_{s\sim p(s_t=s|\rho_0,\mu_\theta)}[A(s)]\\
&=\gamma^0\mathbb E_{s\sim p(s_0=s|\rho_0,\mu_\theta)}[A(s)]+\sum^\infty_{t=1}\gamma^t\mathbb E_{s\sim p(s_t=s|\rho_0,\mu_\theta)}[A(s)]\\
&=\sum^\infty_{t=0}\gamma^t\mathbb E_{s\sim p(s_t=s|\rho_0,\mu_\theta)}[A(s)]\\
&=\mathbb E_{s\sim\rho^{\mu_\theta}}[A(s)]
\end{split}
$$

其中$\rho^{\mu_\theta}$为**折扣状态概率分布**

$$
\rho^{\mu_\theta}(s)=\sum^\infty_{t=0}\gamma^tp(s_t=s|\rho_0,\mu_\theta)
$$

这表明，$Q^{\mu_\theta}(s,\mu_\theta(s))$的梯度在初始状态分布的期望意义下，第二项$B(s)$对梯度的影响可以被合并到第一项$A(s)$的折扣状态分布期望中。由此我们得到一个简洁优美的**确定性策略梯度定理**

$$
\nabla_\theta J(\mu_\theta)=\mathbb E_{s\sim\rho^{\mu_\theta}}[\nabla_\theta\mu_\theta(s)\nabla_aQ^{\mu_\theta}(s,a)|_{a=\mu_\theta(s)}]
$$

## 离轨策略

学习一个确定性策略很容易让我们考虑到探索的问题。在DPG中，出于其独特的策略更新与评估方式，我们不需要太担心动作的探索能力，因为评估动作的$Q$函数由神经网络近似，是动作$a$的平滑函数，具有一定的泛化能力，而策略也根据$Q$函数在$a$处的局部梯度进行更新，不依赖其他动作的采样也能产生改进与探索。

但是动作的探索不能保证状态的探索，尤其是当动作依赖梯度发生小步长的改进时，这样微小的动作变化很可能不会带来显著的状态转移差异，在稀疏奖励、极易失败终止、层次化结构这样的环境中将会失效。如果想进一步提升动作的探索度，就要向动作加入探索噪声，但这会导致$Q$函数收到极其嘈杂的学习信号，策略网络将无法区分某个状态的好坏是由于策略本身的质量还是由于探索噪声的随机影响。

于是DPG选择只提升状态空间的探索性，由向目标策略$\mu(s)$添加噪声$\mathcal N$得到的行为策略$\beta(s)=\mu(s)+\mathcal N$来实现。但是在智能体根据行为策略产生的轨迹中，我们不采样动作序列，只采样状态转移序列。由行为策略来探索状态，而由目标策略在行为策略经历的状态中分别选择动作。

那么，万一有一个状态，对于所有其他状态，执行$a=\mu_\theta(s)$所在的局部梯度更新范围内的任何动作都无法到达，对该状态的探索不就没有意义了吗？这是包括DPG在内的大多数基于动作扰动的探索方法的一个根本局限，只能用更高级的探索策略来解决。

除此之外，DPG通过理想假设回避了行为策略状态采样带来的分布偏差问题，因为在连续状态空间中，状态重要度采样比的估计难度极大。这一问题将在DDPG中通过工程技巧来在一定程度上缓解。

## 网络框架

DPG使用Actor Critic方法的网络框架，其中

- **Actor网络**
  - 输入：状态$s$
  - 输出：确定动作$a=\mu_\theta(s)$
  - 参数更新：$\theta_{t+1}=\theta_t+\alpha_\theta\nabla_\theta\mu_\theta(s)\nabla_aQ^{\mu_\theta}(s,a)|_{a=\mu_\theta(s)}$
- **Critic网络**
  - 输入：状态$s$，动作$a$
  - 输出：动作价值$Q_w(s,a)$
  - 参数更新（TD方法）：$w_{t+1}=w_t-\alpha_w\delta_t\nabla_wQ_w(s_t,a_t)$

DPG算法的主循环如下：

1. 从初始状态$s_0\sim\rho_0$开始，智能体根据行为策略$\beta(s)=\mu_\theta(s)+\mathcal N$生成一段轨迹，并记录状态转移序列$(s_0,s_1,\cdots)$；
2. 对于行为策略经历的每个状态$s_t$，目标策略执行动作$a_t=\mu_\theta(s_t)$，得到奖励$r_t$和后继状态$s'_t$（与行为策略的状态转移序列区分开），再由目标策略采样动作$a_t'=\mu_\theta(s_t')$；
3. Critic计算$Q_w(s_t,a_t)$和$Q_w(s'_t,a'_t)$；
4. Actor用Critic的评价更新目标策略$\theta_{t+1}=\theta_t+\alpha_\theta\nabla_\theta\mu_\theta(s)\nabla_aQ^{\mu_\theta}(s,a)|_{a=\mu_\theta(s)}$（行为策略被动跟随）；
5. Critic用TD误差更新估计$w_{t+1}=w_t-\alpha_w\delta_t\nabla_wQ_w(s_t,a_t)$；
6. 重复上述步骤直至策略和价值估计收敛。

下一篇将介绍DPG的两个改进算法：DDPG和TD3。

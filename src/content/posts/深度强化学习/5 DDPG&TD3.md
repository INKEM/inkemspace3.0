---
title: 【深度强化学习】#5 DDPG&TD3：确定性策略梯度的改进
date: 2025-11-01
summary: DPG方法的工程改进。
tags: [DDPG, TD3, 经验回放, 目标网络, 双学习, 离轨策略]
category: 深度强化学习
---

上一篇我们介绍了DPG算法，其重要贡献在于给出了确定性策略$\mu(s)=a$的策略梯度定理

$$
\nabla_\theta J(\mu_\theta)=\mathbb E_{s\sim\rho^{\mu_\theta}}[\nabla_\theta\mu_\theta(s)\nabla_aQ^{\mu_\theta}(s,a)|_{a=\mu_\theta(s)}]
$$

尽管DPG在理论上是完备的，但当我们采用深度神经网络来近似$\mu(s)$和$Q(s,a)$时，就会遇到深度学习中普遍存在的挑战，例如训练不稳定、收敛缓慢等。DDPG和TD3算法均为从工程实践上解决这些问题而被提出。

# DDPG

**DDPG**（Deep Deterministic Policy Gradient，深度确定性策略梯度）将DQN的思想，经验回放和目标网络，应用到了DPG中。

## 经验回放

经验回放主要用于满足训练数据的独立同分布要求。DDPG的经验回放与DQN中一样，这里仅做回顾。

> 它将经验$(s_t,a_t,r_t,s_{t+1})$存储在一个固定大小的回放缓冲区中。在训练前，先让智能体采用某个行动策略$\pi$与环境持续交互，收集多条经验直至回放缓冲区存满。随后利用该回放缓冲区对Q网络进行训练，训练过程的每一轮等概率随机从缓冲区中抽取一个batch大小的经验训练网络，算出每个经验的梯度后使用梯度的平均更新参数。

## 双目标网络与软更新

在Actor-Critic框架下，DDPG为Actor和Critic分别创建了一个目标网络，即目标Actor网络$\mu'_{\theta'}(s)$和目标Critic网络$Q'_{w'}(s,a)$。在训练时，TD误差中更新目标的计算，即关于后继状态的动作选择和价值评估，均采用稳定的目标网络而非持续更新的主网络

$$
\delta_t=r_t+\gamma\cdot Q_{w'}'(s_{t+1},\mu'_{\theta'}(s_{t+1}))-Q_w(s_t,\mu_\theta(s_t))
$$

Actor网络的策略梯度更新不需要目标网络，因为其并不依赖于自身的估计。

在DQN中，目标网络的更新方式为定期传参，即网络更新一定次数后，其权重更新结果才会复制给目标网络。与之不同，DDPG提出了一种软更新方式，在每一步都对目标网络的参数进行微调

$$
\begin{split}
\theta'\leftarrow\tau\theta+(1-\tau)\theta'\\
w'\leftarrow\tau w+(1-\tau)w'
\end{split}
$$

其中$\tau<<1$，例如取$0.01$。

这避免了定期传参带来的目标网络参数跳跃式变化，进一步稳定了学习过程。

## 离轨策略

原始的DPG算法在对探索性的提升上，仅仅让带噪声的行为策略$\beta(s)=\mu(s)+\epsilon$产生状态转移序列，而计算所用到的动作$a$则全部由目标策略$\mu(s)$给出，这是考虑到了动作噪声会让$Q(s,a)$的评估脱离目标策略，进而影响策略梯度的改进效果。

但是让我们重新审视动作价值函数的期望表示

$$
q_\pi(s,a)=\mathbb E[R_{t+1}+\gamma v_\pi(S_{t+1})|S_t=s,A_t=a]
$$

（_传统强化学习理论以转移到新状态为考量，即时收益下标为t+1；而现代深度强化学习在实践中以同一个时间步输入动作-状态输出即时收益的对应关系为考量，其下标为t_）

可以看到，策略$\pi$对$q(s,a)$的影响仅存在于后继状态的动作选择，而智能体在当下选择的动作$A_t=a$仅作为一个变量被传入$q(s,a)$中，决定即时收益$R_t$。这意味着，如果我们传入行为策略$\beta(s)$的选择作为$A_t$，而$A_{t+1}$仍由目标策略$\mu(s)$给出，不仅不影响$Q(s,a)$对目标策略动作选择的评估，还能探索到其他动作的价值，推动策略梯度朝着更优的方向改进。

由此，DDPG最终在更新Critic网络时使用的TD误差计算如下

$$
\delta_t=r_t+\gamma\cdot Q_{w'}'(s_{t+1},\mu'_{\theta'}(s_{t+1}))-Q_w(s_t,\beta(s_t))
$$

## 算法流程

向DPG加入经验回放和目标网络两大技术后，我们得到DDPG的算法流程如下：

1. 初始化
   1. Actor网络$\mu_\theta(s)$和Critic网络$Q_w(s,a)$；
   2. 目标Actor网络$\mu'_{\theta'}(s)$和目标Critic网络$Q'_{w'}(s,a)$，参数分别从主网络复制而来；
   3. 经验回放缓存区$R$；
   4. 预设经验回放的批量大小$N$。
2. 循环（每一幕）
   1. 初始化噪声$\epsilon$用于探索；
   2. 初始化状态$s_1$；
   3. 循环（每个时间步$t$）
      1. 根据带噪声策略选择动作$a_t=\mu_\theta(s)+\epsilon_t$并执行，得到奖励$r_t$和后继状态$s_{t+1}$；
      2. 将转移样本$(s_t,a_t,r_t,s_{t+1})$存入回放缓冲区$R$；
      3. 如果样本数量$|R|\geqslant N$，则随机采样一个小批量的$N$个转移样本$(s_i,a_i,r_i,s_{i+1})$；
      4. 计算样本批量的平均TD误差梯度更新Critic网络$$w_{t+1}=w_t-\alpha_w\frac1N\sum_{i=1}^N(\delta_i\nabla_wQ_w(s_i,a_i))$$
      5. 计算样本批量的平均策略梯度更新Actor网络$$\theta_{t+1}=\theta_t+\alpha_\theta\frac1N\sum_{i=1}^N(\nabla_\theta\mu_\theta(s_i)\nabla_aQ_{w}(s_i,a)|_{a=\mu_\theta(s)})$$
      6. 软更新目标网络$$\begin{split}\theta'\leftarrow\tau\theta+(1-\tau)\theta'\\w'\leftarrow\tau w+(1-\tau)w'\end{split}$$
3. 直到Actor网络和Critic网络收敛。

# TD3

**TD3**（Twin Delayed Deep Deterministic Policy Gradient，双延迟深度确定性策略梯度）在DDPG的基础上，缓解了Q值高估问题，同时进一步增强了其训练过程的稳定性。

## 双Critic网络

在DDPG中，触发高估问题的最大化操作没有显式地表现，而是隐含在Actor网络的训练目标中，即最大化$Q(s,\mu(s))$。这意味着Actor会主动去寻找并利用Critic的高估区域，这一误差也将在软更新中被延迟传递到目标Actor网络用于更新Critic网络。

TD3应对这一问题的方法依旧采用了双学习的思想，即维护两个Critic网络$Q_{w_1}(s,a)$和$Q_{w_2}(s,a)$，以及它们的目标网络$Q_{w_1'}'(s,a)$和$Q_{w_2'}'(s,a)$。在计算Critic网络的更新目标时，我们采用取最小值操作降低高估误差

$$
y_t=r_t+\gamma\min{}(Q_{w_1'}'(s_{t+1},a_{t+1}),Q_{w_2'}'(s_{t+1},a_{t+1}))
$$

相应地对于每个主网络$Q_{w_j}(s,a)$，其TD误差为

$$
\delta_t=r_t+\gamma\min{}(Q_{w_1'}'(s_{t+1},a_{t+1}),Q_{w_2'}'(s_{t+1},a_{t+1}))-Q_{w_j}(s_t,a_t)
$$

在计算策略梯度时，我们仍只使用其中一个Critic网络$Q_{w_1}(s,a)$，另一个网络$Q_{w_2}(s,a)$仅用于稳定更新目标。

## 目标策略平滑

我们通常认为，真实的Q函数应该是平滑的，这源于物理世界普遍的连续性和机器学习的泛化目标。而实际训练过程中，训练数据的不足和噪声的影响可能会导致Critic网络过拟合到不平滑的Q函数，从而对动作的微小变化极其敏感。

TD3通过对计算更新目标所采用的动作添加噪声来平滑Q函数。具体而言，对于原始的更新目标计算公式

$$
y_t=r_t+\gamma\min{}(Q_{w_1'}'(s_{t+1},a_{t+1}),Q_{w_2'}'(s_{t+1},a_{t+1}))
$$

我们不使用目标策略网络的原始输出$a_{t+1}=\mu_\theta(s_{t+1})$，而是经过了平滑的版本

$$
\begin{split}
a_{t+1}=\mu_\theta(s_{t+1})+\epsilon'\\
\epsilon'\sim\mathrm{clip}(\mathcal N,-c,c)
\end{split}
$$

其中$\epsilon'$为平滑噪声，它可以由高斯噪声$\mathcal N$经裁剪限制在范围$[-c,c]$得到。

由此，如果在对某个需要更新的$Q(s_t,a_t)$计算$y_t$时，原始的$Q(s_{t+1},\mu(s_{t+1}))$是一个尖峰，那么通过添加平滑噪声，$a_{t+1}$将很可能采样到周围骤降的位置，从而避开了尖峰对$Q(s_t,a_t)$更新的影响，而高斯噪声的正态分布和裁剪操作又能避免更新目标偏离原始动作。

可见目标策略平滑是一种防御性方法，它可以防止尖峰错误传播，但本身并不主动修复已经存在的尖峰。而尖峰的消除则是由双Critic网络、小批量梯度下降和下面介绍的延迟策略更新，在这几种方法的相互作用下被动消除的。

## 延迟策略更新

在训练初期，Critic网络的误差通常很大，如果此时频繁地更新Actor网络，将会导致Actor朝着一个错误的方向快速变化，使其训练过程极不稳定。延迟策略更新会等待Critic网络在已经经过了足够多次的更新，价值估计相对准确之后，再更新Actor网络，例如每更新两次Critic网络后更新一次Actor网络。

## 算法流程

现在，我们有TD3的算法流程如下：

1. 初始化
   1. Actor网络$\mu_\theta(s)$和Critic网络$Q_{w_1}(s,a)$，$Q_{w_2}(s,a)$；
   2. 目标Actor网络$\mu_{\theta'}'(s)$和目标Critic网络$Q_{w_1'}'(s,a)$，$Q_{w_2'}'(s,a)$，参数分别从主网络复制而来；
   3. 经验回放缓存区$R$；
   4. 超参数：预设经验回放的批量大小$N$，Actor网络更新间隔$n$，平滑噪声$\epsilon'\sim\mathrm{clip}(\mathcal N,-c,c)$。
2. 循环（每一幕）
   1. 初始化噪声$\epsilon$用于探索；
   2. 初始化状态$s_1$；
   3. 循环（每个时间步$t$）
      1. 根据带噪声策略选择动作$a_t=\mu_\theta(s_t)+\epsilon_t$，得到奖励$r_t$和后继状态$s_{t+1}$；
      2. 将转移样本$(s_t,a_t,r_t,s_{t+1})$存入回放缓冲区$R$；
      3. 如果样本数量$|R|\geqslant N$，则随机采样一个小批量的$N$个转移样本$(s_i,a_i,r_i,s_{i+1})$；
      4. 对每个转移样本$(s_i,a_i,r_i,s_{i+1})$计算目标动作$a_{i+1}=\mu_\theta(s_{i+1})+\epsilon'$；
      5. 计算样本批量的平均TD误差梯度更新每个Critic网络$$w_{t+1}^{(j)}=w_t^{(j)}-\alpha_{w^{(j)}}\frac1N\sum_{i=1}^N(\delta_i\nabla_{w^{(j)}}Q_{w^{(j)}}(s_i,a_i))$$
      6. Critic网络每更新$n$次后，计算样本批量的平均策略梯度更新Actor网络$$\theta_{t+1}=\theta_t+\alpha_\theta\frac1N\sum_{i=1}^N(\nabla_\theta\mu_\theta(s_i)\nabla_aQ_{w_1}(s_i,a)|_{a=\mu_\theta(s)})$$
      7. 软更新目标网络$$\begin{split}\theta'\leftarrow\tau\theta+(1-\tau)\theta'\\w_j'\leftarrow\tau w_j+(1-\tau)w_j'\end{split}$$
3. 直到Actor网络和Critic网络收敛。

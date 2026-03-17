---
title: 【深度强化学习】#7 SAC：最大熵与重参数化技巧
date: 2025-11-04
summary: 深度强化学习的集大成者。
tags: [SAC, 熵正则化, 策略梯度, 重参数化]
category: 深度强化学习
---

至此，Actor-Critic方法的发展走过了DDPG/TD3和TRPO/PPO两条技术路线。DDPG/TD3具有准确的价值估计，但确定性策略这一核心使得智能体的探索严重依赖人为添加到动作的外部噪声，既引入了更繁琐的超参数调优，又从本质上限制了智能体在复杂环境中进行多样化探索的能力；TRPO/PPO既稳定了策略更新，又保证了其探索性，但在价值估计上的优化却略显不足。二者各有优劣又相互互补，于是集两家之所长的SAC算法应运而生。

# SAC

**SAC**（Soft Actor-Critic，软演员-评论家）的核心突破在于将PPO中的最大熵原则引入强化学习的目标中，要求智能体在追求累积回报的同时尽可能保持策略的随机性，同时又继承了TD3在价值估计中的技术精华，最终在算法性能上实现了对这两类方法的显著超越。

## 最大熵强化学习

对于原始的策略梯度，其目标是最大化期望累积奖励

$$
J(\theta)=\mathbb E_{(s_t,a_t)\sim\rho_\pi}\left[\sum^T_{t=0}\gamma^tr(s_t,a_t)\right]
$$

而SAC的目标是在最大化累积奖励的同时，最大化策略的熵，以鼓励探索

$$
J(\theta)=\mathbb E_{(s_t,a_t)\sim\rho_\pi}\left[\sum^T_{t=0}\gamma^t(r(s_t,a_t)+\alpha\mathcal H(\pi_\theta(\cdot|s_t)))\right]
$$

其中$\alpha$为温度系数，控制熵的重要程度。温度系数最简单的设置方法即手动调优，根据不同问题下策略的收敛速度和随机性设定。现代SAC常用自动熵调整，通过最小化损失函数$L(\alpha)=\alpha(\log\pi(a|s)+\tilde{\mathcal H})$学习，其中$\tilde{\mathcal H}$是人为设定的目标熵，根据经验通常设为$-\mathrm{dim}(\mathcal A)$，即负的动作空间维度，以鼓励策略在每个动作维度上接近均匀分布。

对于新的策略评估方式，SAC也定义了新的Q函数，称为soft Q函数

$$
Q_{\mathrm{soft}}^\pi(s_t,a_t)\dot=\mathbb E_{(s_t,a_t)\sim\rho_\pi}\left[r(s_t,a_t)+\sum^T_{k=1}\gamma^k(r(s_{t+k},r_{t+k})+\alpha\mathcal H(\pi(\cdot|s_{t+k})))\right]
$$

soft Q函数将策略的熵同样视为一种奖励，融入在$s_t$执行动作$a_t$后能获得的期望累积回报当中，而Critic网络也将直接学习如何给出soft Q函数的估计值。soft Q函数不包含$\alpha\mathcal H(\pi(\cdot|s_{t}))$，是因为它由$s_t$决定而与$a_t$无关。

现在，我们可以将目标函数写为

$$
J(\theta)=\mathbb E_{s\sim\rho_\pi,a\sim\pi_\theta}\left[Q^{\pi}_{\mathrm{soft}}(s,a)+\alpha\mathcal H(\pi_\theta(\cdot|s))\right]
$$

（_关于目标函数的状态-动作是否带下标t，这取决于我们是从整条轨迹的视角，还是从静态分布的视角来理解。通常轨迹视角使用求和形式，而静态分布视角使用价值函数形式_）

根据策略梯度定理的一般形式，我们有目标函数的梯度

$$
\nabla_\theta J(\theta)=\mathbb E_{s\sim\rho_\pi,a\sim\pi_\theta}\left[\nabla_\theta\ln\pi_\theta(a|s)Q^{\pi}_{\mathrm{soft}}(s,a)]+\mathbb E_{s\sim\rho_\pi}[\nabla_\theta\alpha\mathcal H(\pi_\theta(\cdot|s))\right]
$$

但是该梯度计算具有较大的方差，SAC对其进行了优化。

## 重参数化技巧

SAC利用重参数化技巧从动作采样这一根源上降低了方差。

在DPG一章中，我们推导出确定性策略梯度定理如下

$$
\nabla_\theta J(\theta)=\mathbb E_{s\sim\rho^{\mu_\theta}}[\nabla_\theta\mu_\theta(s)\nabla_aQ^{\mu_\theta}(s,a)|_{a=\mu_\theta(s)}]
$$

但是在Actor-Critic一章中，同样对于Q函数，我们使用的随机性策略梯度定理为

$$
\nabla_\theta J(\theta)=\mathbb E_{s\sim\rho_\pi,a\sim\pi_\theta}[\nabla_\theta\ln\pi_\theta(a|s)Q(s,a)]
$$

对比发现，之前的随机性策略算法只关注了$\theta$对$\pi_\theta$的梯度，却没有利用$\theta\rightarrow\pi_\theta\rightarrow a\rightarrow Q(s,a)$这条路径的梯度。这是因为在随机性策略梯度的推导中，我们用采样得到的回报$G(\tau)$代替了真实函数$Q(s,a)$，二者在期望意义下是等价的。如果直接计算$Q(s,a)$对$a$的梯度，我们就无法消去环境动态特性$p(s_{t+1}|s_t,a_t)$，进而无法将梯度算符$\nabla$移到期望中，而环境动态特性通常是未知的。但是放弃利用动作梯度的信息而仅仅依赖采样拟合的代价就是高方差。

重参数化技巧通过规范策略的参数化形式，切断了采样随机性和策略网络参数$\theta$之间的关系，打通了一条完全确定的梯度传播路径。我们通常采用高斯分布作为策略的形式，因为这符合我们对连续控制任务的直觉，也易于数学上的计算和随机性的调节。此时动作的分布为

$$
a\sim\mathcal N(\mu_\theta(s),\sigma_\theta(s)^2)
$$

策略网络将以状态$s$为输入，输出高斯分布的均值$\mu$和标准差$\sigma$。而为了让随机性与参数$\theta$无关，我们变换该分布如下

$$
\begin{split}
&a=f_\theta(\epsilon;s)=\mu_\theta(s)+\sigma_\theta(s)\cdot\epsilon&\\
&\quad\epsilon\sim\mathcal N(0,I)
\end{split}
$$

由此，我们不再直接采样动作$a$，而是先从一个固定的、与参数$\theta$无关的基础分布（如标准正态分布$\mathcal N(0,1)$）中采样一个随机噪声$\epsilon$，再通过一个确定的、可微的变换构造出动作$a$。

## 策略梯度推导

结合重参数化技巧，我们重新得到SAC的目标函数为

$$
\begin{split}
J(\theta)&=\mathbb E_{s\sim\rho_\pi,\epsilon\sim\mathcal N}\left[Q^{\pi}_{\mathrm{soft}}(s,f_\theta(\epsilon;s))+\alpha\mathcal H(\pi_\theta(f_\theta(\epsilon;s)|s))\right]\\
&=\mathbb E_{s\sim\rho_\pi,\epsilon\sim\mathcal N}\left[Q^{\pi}_{\mathrm{soft}}(s,f_\theta(\epsilon;s))-\alpha\log\pi_\theta(f_\theta(\epsilon;s)|s)\right]
\end{split}
$$

需要辨析的是，$f_\theta(\epsilon;s)$和$\pi_\theta$都源于同一个策略，区别在于$f_\theta(\epsilon;s)$是一个动作生成器，负责采样；而$\pi_\theta$是一个概率分布函数，负责评估动作的概率。

现在SAC的策略梯度为

$$
\begin{split}
\nabla_\theta J(\theta)&=\nabla_\theta\mathbb E_{s\sim\rho_\pi,\epsilon\sim\mathcal N}\left[Q^{\pi}_{\mathrm{soft}}(s,f_\theta(\epsilon;s))-\alpha\log\pi_\theta(f_\theta(\epsilon;s)|s)\right]\\
&=\mathbb E_{s\sim\rho_\pi,\epsilon\sim\mathcal N}\left[\nabla_\theta(Q^{\pi}_{\mathrm{soft}}(s,f_\theta(\epsilon;s))-\alpha\log\pi_\theta(f_\theta(\epsilon;s)|s))\right]
\end{split}
$$

Soft Q函数的梯度根据链式法则有

$$
\nabla_\theta(Q^{\pi}_{\mathrm{soft}}(s,f_\theta(\epsilon;s)))=\nabla_aQ^\pi_{\mathrm{soft}}(s,a)|_{a=f_\theta(\epsilon;s)}\cdot\nabla_\theta f_\theta(\epsilon;s)
$$

熵的梯度

$$
\nabla_\theta(-\alpha\log\pi_\theta(f_\theta(\epsilon;s)|s))=-\alpha\nabla_\theta\log\pi_\theta(f_\theta(\epsilon;s)|s)
$$

$\log\pi_\theta(f_\theta(\epsilon;s)|s)$是$a$和$\theta$的函数，同样根据链式法则有

$$
\nabla_\theta\log\pi_\theta(f_\theta(\epsilon;s)|s)=\nabla_\theta\log\pi_\theta(a|s)|_{a=f_\theta(\epsilon;s)}+\nabla_a\log\pi_\theta(a|s)|_{a=f_\theta(\epsilon;s)}\cdot\nabla_\theta f_\theta(\epsilon;s)
$$

其中第一项是保持动作$a$不变时，策略概率关于$\theta$的梯度；第二项是$\theta$引起动作$a$改变对策略概率的梯度。

第一项梯度对于任意给定的状态$s$期望为零，证明如下

$$
\begin{split}
\mathbb E_{a\sim\pi_\theta}[\nabla_\theta\log\pi_\theta(a|s)]&=\int\pi_\theta(a|s)\cdot\nabla_\theta\log\pi_\theta(a|s)\mathrm da\\
\text{（自然对数导数性质）}&=\int\pi_\theta(a|s)\cdot\frac{\nabla_\theta\pi_\theta(a|s)}{\pi_\theta(a|s)}\mathrm da\\
&=\int\nabla_\theta\pi_\theta(a|s)\mathrm da\\
\text{（梯度变量与积分变量不一致时，梯度算符可提出）}&=\nabla_\theta\int\pi_\theta(a|s)\mathrm da\\
\text{（概率分布的归一化条件）}&=\nabla_\theta1=0
\end{split}
$$

最终我们得到SAC完整的策略梯度如下

$$
\nabla_\theta J(\theta)=\mathbb E_{s\sim\rho_\pi,\epsilon\sim\mathcal N}\left[(\nabla_aQ^\pi_{\mathrm{soft}}(s,a)|_{a=f_\theta(\epsilon;s)}-\alpha\nabla_a\log\pi_\theta(a|s)|_{a=f_\theta(\epsilon;s)})\cdot\nabla_\theta f_\theta(\epsilon;s)\right]
$$

随机梯度更新采用的梯度则为

$$
\nabla_\theta J(\theta)=(\nabla_aQ^\pi_{\mathrm{soft}}(s,a)|_{a=f_\theta(\epsilon;s)}-\alpha\nabla_a\log\pi_\theta(a|s)|_{a=f_\theta(\epsilon;s)})\cdot\nabla_\theta f_\theta(\epsilon;s)
$$

## 网络框架

在其他技术细节上，SAC继承了TD3算法。网络框架上，SAC拥有一个Actor网络和双Critic-Q网络，以及它们各自的目标网络，并且目标网络的更新方式为软更新。但研究表明TD3中的延迟策略更新在SAC中效果并不显著，甚至会减慢策略的适应速度进而降低性能，因此没有被采用。

每个Q网络$Q_{w_j}(s,a)$的目标值为（由目标网络计算）

$$
y_t=r(s_t,a_t)+\gamma\left(\min_{j=1,2}Q'_{w_j'}(s_{t+1},f'_{\theta'}(\epsilon;s_{t+1}))-\alpha\log\pi'_{\theta'}(f'_{\theta'}(\epsilon;s_{t+1})|s_{t+1})\right)
$$

Actor网络的策略梯度为（由主网络计算）

$$
\nabla_\theta J(\theta)=(\nabla_aQ_{w_1}(s,a)|_{a=f_\theta(\epsilon;s)}-\alpha\nabla_a\log\pi_\theta(a|s)|_{a=f_\theta(\epsilon;s)})\cdot\nabla_\theta f_\theta(\epsilon;s)
$$

原始的SAC算法还使用了一个Critic-V网络，它用于估计Q网络除了$r(s_t,a_t)$的部分，进而计算Q网络的目标值

$$
y=r(s_t,a_t)+\gamma V(s_{t+1})
$$

而双Q网络则用于计算V网络的目标值。但这仅仅是为了构建贝尔曼方程的形式，实现理论上的优雅。后来作者发现V网络是多余的，移除V网络可以降低算法复杂度、避免更多估计误差，算法性能却相当甚至更好。

## 算法流程

SAC的算法流程如下：

1. 初始化
   1. Actor网络$\pi_\theta(a|s)$和Critic网络$Q_{w_1}(s,a)$，$Q_{w_2}(s,a)$；
   2. 目标Actor网络$\pi_{\theta'}'(a|s)$和目标Critic网络$Q_{w_1'}'(s,a)$，$Q_{w_2'}'(s,a)$，参数分别从主网络复制而来；
   3. 经验回放缓存区$R$；
   4. 超参数：温度系数$\alpha$，动作选择器噪声$\epsilon\sim\mathcal N(0,I)$。
2. 循环（每一幕）
   1. 初始化状态$s_1$；
   2. 循环（每个时间步$t$）
      1. 由动作选择器（主网络）采样动作$a_t=f_\theta(\epsilon;s_t)$，得到奖励$r_t$和后继状态$s_{t+1}$；
      2. 将转移样本$(s_t,a_t,r_t,s_{t+1})$存入回放缓冲区$R$；
      3. 如果样本数量$|R|\geqslant N$，则随机采样一个小批量的$N$个转移样本$(s_i,a_i,r_i,s_{i+1})$；
      4. 对每个转移样本$(s_i,a_i,r_i,s_{i+1})$，由动作选择器（目标网络）采样下一个动作$a_i'=f'_{\theta'}(\epsilon;s_{i+1})$，并计算其对数概率$\log\pi'_{\theta'}(a_i'|s_{i+1})$；
      5. 计算样本批量的平均梯度更新每个Critic网络$$w_{t+1}^{(j)}=w_t^{(j)}-\alpha_{w^{(j)}}\frac1N\sum_{i=1}^N[(y_i-Q_{w^{(j)}}(s_i,a_i))\cdot\nabla_{w^{(j)}}Q_{w^{(j)}}(s_i,a_i)]$$
      6. 计算样本批量的平均梯度更新Actor网络$$\theta_{t+1}=\theta_t+\alpha_\theta\frac1N\sum_{i=1}^N\nabla_\theta J_i(\theta)$$
      7. 软更新目标网络$$\begin{split}\theta'\leftarrow\tau\theta+(1-\tau)\theta'\\w_j'\leftarrow\tau w_j+(1-\tau)w_j'\end{split}$$
3. 直到Actor网络和Critic网络收敛。

---
title: 【深度强化学习】#3 Actor-Critic方法基础与A3C/A2C
date: 2025-10-12
summary: 基于策略的方法与基于价值的方法相结合。
tags: [Actor-Critic, A3C, A2C]
category: 深度强化学习
---

尽管在上一篇末尾，我们向REINFORCE引入了基线来缓解$G_t$带来的高方差，但仍然没有触及问题的核心，即$G_t$采样中的随机性。如果将采样值$G_t$替换为期望值$Q(s_t,a_t)$或$V(s_t,a_t)$，就能显著降低方差。期望值作为价值函数则可以由价值函数近似来给出。

**Actor-Critic**算法将策略梯度与价值函数近似相结合。其中策略函数是Actor（演员），负责根据当前的状态决定动作；价值函数是Critic（评论家），负责在演员做完动作后根据环境反馈评价这个动作有多好。

# QAC

**QAC**是最简单的Actor-Critic算法，它直接使用动作价值函数$Q_w(s_t,a_t)$来代替$G_t$作为策略函数的更新依据（本系列同时使用价值函数与策略函数时，价值函数的参数记为$w$，策略函数的参数记为$\theta$）

$$
\theta_{t+1}=\theta_t+\alpha\nabla_\theta\ln\pi_\theta(a_t|s_t)\cdot Q_w(s_t,a_t)
$$

在QAC中，Q网络的更新目标不再基于Q学习而是SARSA，因为Q学习是离线策略的，评估的是贪心策略而非Actor执行的探索策略。基于SARSA的Q网络的估计目标是在状态$s_t$下执行动作$a_t$，并且之后都遵循当前策略$\pi_\theta$的期望累计回报，即标签为

$$
y_t=r_t+\gamma\cdot Q_w(s_{t+1},a_{t+1})
$$

网络输出与标签的误差即**TD误差**

$$
\delta_t=r_t+\gamma\cdot Q_w(s_{t+1},a_{t+1})-Q_w(s_t,a_t)
$$

则损失函数为

$$
L=\frac12\delta_t^2=\frac12[r_t+\gamma\cdot Q_w(s_{t+1},a_{t+1})-Q_w(s_t,a_t)]^2
$$

参数更新公式为

$$
w_{t+1}=w_t-\alpha\delta_t\cdot\nabla_wQ_w(s_t,a_t)
$$

于是QAC算法的完整流程如下：

1. Actor根据当前状态$s_t$和策略$\pi_\theta$选择动作$a_t$；
2. 环境反馈奖励$r_t$和新状态$s_{t+1}$；
3. 在新状态$s_{t+1}$下根据策略$\pi_\theta$采样下一个动作$a_{t+1}$；
4. Critic计算$Q_w(s_t,a_t)$和$Q_w(s_{t+1},a_{t+1})$；
5. Actor用Critic的评价更新策略$\theta_{t+1}=\theta_t+\alpha\nabla_\theta\ln\pi_\theta(a_t|s_t)\cdot Q_w(s_t,a_t)$；
6. Critic用TD误差更新估计$w_{t+1}=w_t-\beta\delta_t\nabla_wQ_w(s_t,a_t)$；
7. 重复上述步骤直至策略和价值估计收敛。

# Advantage Actor Critic

同样地，Actor-Critic算法也可以引入基线来过滤状态本身价值高低带来的波动。**Advantage Actor Critic**（本文简称AAC，但实际上没有缩写用于指代这一原始算法，网上许多文章对它的谬称A2C其实是A3C之后才出现的变体，详见后文）采取优势函数作为策略函数$A(s,a)=Q(s,a)-V(s)$的更新依据

$$
\theta_{t+1}=\theta_t+\alpha\nabla_\theta\ln\pi_\theta(a_t|s_t)\cdot A_w(s_t,a_t)
$$

但是优势函数包含$Q$和$V$两种函数，同时学习二者很麻烦，因此AAC利用了一个关键等式

$$
Q(s_t,a_t)=\mathbb E[r_t+\gamma\cdot V(s_{t+1})]
$$

由此优势函数就可以只用一个$V$网络和采样奖励来估计

$$
\begin{split}
A_w(s_t,a_t)&=\mathbb E[r_t+\gamma\cdot V_w(s_{t+1})]-V_w(s_t)\\
&\approx r_t+\gamma\cdot V_w(s_{t+1})-V_w(s_t)
\end{split}
$$

对优势函数的单次采样正是状态价值函数的TD误差$\delta_t$，因此AAC和QAC的参数更新可写成相似的形式

$$
w_{t+1}=w_t-\alpha\delta_t\cdot\nabla_wV_w(s_t)
$$

AAC算法的完整流程如下：

1. Actor根据当前状态$s_t$和策略$\pi_\theta$选择动作$a_t$；
2. 环境反馈奖励$r_t$和新状态$s_{t+1}$；
3. Critic计算$V_w(s_t)$和$V_w(s_{t+1})$；
4. Actor用Critic的评价更新策略$\theta_{t+1}=\theta_t+\alpha A_w(s_t,a_t)\cdot\nabla_\theta\ln\pi_\theta(a_t|s_t)$；
5. Critic用TD误差更新估计$w_{t+1}=w_t-\beta\delta_t\cdot\nabla_wV_w(s_t)$；
6. 重复上述步骤直至策略和价值估计收敛。

尽管在AAC中，$A_w(s_t,a_t)$的计算方式就是TD误差$\delta_t$，但为了阐明其背后的意义，即Actor最大化优势函数以优化动作选择，Critic最小化TD误差以使估计更准确，在更新公式中不会将二者混为一谈。

# A3C与A2C

在DQN中，我们提到经验相关的问题，即一个智能体连续与环境交互得到的数据存在相关性。DQN使用经验回放来解决，但仍然存在一些局限性：

- 需要大量内存存储经验。
- 采取离线学习，为了重复利用数据和稳定训练将所有旧策略的数据均用于学习新策略，存在滞后性，且新策略数据的加入会使得数据分布不平稳。
- 对环境的探索效率有上限，不主动产生新的探索行为，探索由行为策略决定。

**A3C**（Asynchronous Advantage Actor Critic）在AAC的基础之上，为了解决经验相关问题，提出了**数据并行化**。A3C允许多个智能体并行运行在多个环境实例中，网络分为**全局共享网络**和每个智能体的**本地网络**两种。每个智能体的工作流程如下：

1. 从全局网络同步最新的参数$\Theta,W$到自己的本地网络$\theta,w$；
2. 使用本地网络保持参数不变在自己的环境实例中独立运行$n$步，或遇到终止状态$s_T$提前终止，并对每一步记录$(s_t,a_t,r_t)$；
3. 如果最后一步不是终止状态，则使用本地Critic计算$R_n=V_w(s_{t_n})$，否则$R_T=0$；
4. 从最后一步到第一步反向遍历：
   1. 计算当前步的累积回报$R_t=r_t+\gamma R_{t+1}$；
   2. 计算当前步的优势$A(s_t,a_t)=R_t-V_w(s_t)$；
   3. 累积Critic梯度$g_w\leftarrow g_w+[R_t-V_w(s_t)]\cdot\nabla_wV_w(s_t)$；
   4. 累积Actor梯度$g_\theta\leftarrow g_\theta+A(s_t,a_t)\cdot\nabla_\theta\pi_\theta(a_t|s_t)$；
5. 将累积梯度异步地上传到全局网络并更新：
   1. $\Theta\leftarrow\Theta+\alpha g_\theta$
   2. $W\leftarrow W+\beta g_w$
6. 重复上述步骤直至全局网络的策略和价值估计收敛。

A3C的论文使用$n$步回报的TD误差来估计优势函数。相比单步TD，它依赖于更少的价值函数估计，偏差更小；相比蒙特卡洛，它只往后采样$n$步回报，方差更可控。

A3C的异步在于每个智能体一旦完成一次遍历即可更新全局网络参数，无需等待其他智能体做好更新准备。在A3C提出数据并行化后，人们在实践中发现同步更新比异步更新更好用，于是多智能体并行的、同步的Advantage Actor Critic算法，即**A2C**成为了主流，而原始的单智能体的AAC几乎不再被使用，因此在学术界没有简称。

在A2C中，有一个主进程用于控制所有智能体，其流程如下：

1. 主进程向所有智能体广播全局网络的参数；
2. 主进程等待所有智能体都完成接收后，广播指令使所有智能体同时开始一个批次的训练；
3. 所有智能体都完成运行后，计算出本地累积梯度发送给主进程；
4. 主进程接收到所有智能体的累积梯度后，计算它们的平均梯度来一次性对全局网络进行更新；
5. 重复上述步骤直至全局网络的策略和价值估计收敛。

同步机制让所有的智能体在同一时刻都拥有统一的起点和目标，让全局网络能够得到统一的更新，避免了使用过时的梯度来更新一个已经进化的模型（梯度冲突），使得训练曲线更加平滑稳定，对随机性的敏感度更低。

---
title: 【基于模型的强化学习】#3 MBPO：基于模型的策略优化
date: 2026-02-09
summary: 量化策略在模型下和在真实环境下的性能差距，指导如何利用模型有效地训练策略。
tags: [MBRL, MBPO, 单调改进定理, 分支推演]
category: 基于模型的强化学习
---

# MBPO

上一篇介绍的PETS不依赖显式策略进行决策，主要从规划的角度降低模型误差。规划算法本身具有一定的稳定性，如果一次规划给出的动作序列不好，下一步可以重新规划。并且每次规划都从真实的状态开始探索周围的局部区域，使偏移能够得到限制和纠正。但是在线实时规划需要大量的计算资源，且在高维连续空间中难以进行有效的规划，而显式的策略网络在这些方面具有强大的优势。

然而一旦MBRL依赖于显式的参数化策略，模型作为训练策略的数据生成器将面临比用于规划更大的挑战。模型误差会训练出次优策略，次优策略与真实环境交互得到分布偏移的数据，使用分布偏移的数据更新模型又会带来更大的误差，形成了恶性循环。

**基于模型的策略优化**（Model-Based Policy Optimization，MBPO，2019）通过单调改进定理量化了策略在模型中和在真实环境中的性能差异，为如何有效地利用模型生成数据训练策略提供了理论指导，并且基于这一指导提出了分支推演思想。

## 单调改进定理

在策略优化背景下，MBRL的算法框架如下

1. 初始化策略$\pi(a|s)$，模型$p_\theta(s',r|s,a)$和空数据集$\mathcal D$；
2. 循环$N$次：
   1. 使用策略$\pi$在真实环境中收集数据：$$\mathcal D=\mathcal D\cup\{(s_i,a_i,s'_i.r_i)\}_i$$
   2. 使用数据集$\mathcal D$通过最大似然估计训练模型$p_\theta$：$$\theta\leftarrow\underset\theta{\arg\max}\mathbb E_{\mathcal D}[\ln p_\theta(s',r|s,a)]$$
   3. 使用模型$p_\theta$优化策略$\pi$（$\pi'$表示正在被优化的候选策略，$\hat\eta[\pi']$表示其在模型中的期望累积回报）：$$\pi\leftarrow\underset{\pi'}{\arg\max}\hat\eta[\pi']$$

但是由于模型误差的影响，最大化策略在模型中的期望累积回报并不意味着提升策略在真实环境下的期望累积回报。因此我们需要对这两个指标之间的差异进行分析。

MBPO将策略优化中最大化期望累积回报的目标记为$\eta$

$$
\pi^*=\underset\pi{\arg\max}\eta[\pi]=\underset\pi{\arg\max}\mathbb E_\pi\left[\sum^\infty_{t=0}\gamma^tr(s_t,a_t)\right]
$$

MBPO首先证明了含模型偏差的**单调改进**（Monotonic Improvement）定理，为MBRL方法无需最优模型也能提升策略在真实环境下的表现提供了理论依据。单调改进定理证明的目标形式如下

$$
\eta[\pi]\geq\hat\eta[\pi]-C
$$

其中$\eta[\pi]$表示策略$\pi$在真实环境中的期望累积回报，$\hat\eta[\pi]$表示策略$\pi$在模型中的期望累积回报。假设当前模型是基于$\pi_D$收集的数据训练的，则这个不等式意味着如果我们在模型中找到一个策略$\pi$，使得$\hat\eta[\pi]$比策略$\pi_D$在模型中的表现$\hat\eta[\pi_D]$高出至少$C$，就可以保证$\pi$在真实环境中的表现$\eta[\pi]$一定优于$\pi_D$。因为

$$
\eta[\pi]\geq\hat\eta[\pi]-C\geq(\hat\eta[\pi_D]+C)-C=\hat\eta[\pi_D]\approx\eta[\pi_D]
$$

末尾的$\hat\eta[\pi_D]\approx\eta[\pi_D]$基于一个合理假设：模型上一步是在$\pi_D$的状态分布上训练的，所以在$\pi_D$上的预测应该相对准确。

现在，设计模型的关键便是对真实回报和模型回报之间的差距最大值，即模型的**性能差距上界**（Performance Bound）$C$进行量化。MBPO指出，$C$可以用模型的两个误差来表示

- **泛化误差**$\epsilon_m$：模型本身源于训练数据有限的固有误差；
- **分布偏移误差**$\epsilon_\pi$：新策略访问旧策略未访问的状态区域放大了模型的泛化误差。

泛化误差$\epsilon_m$由**概率近似正确**（Probably Approximately Correct，PAC）理论中的标准**PAC泛化界**（PAC Generalization Bounds）量化

$$
\epsilon_m=\max_t\mathbb E_{s\sim\pi_{D,t}}[D_{TV}(p(s',r|s,a)\|p_\theta(s',r|s,a))]
$$

其中$D_{TV}(\cdot\|\cdot)$为**总变分距离**（Total Variation Distance，TVD），用于衡量两个概率分布之间的差异

$$
D_{TV}(P\|Q)=\frac12\sum_{x\in\mathcal X}|p(x)-q(x)|
$$

$\epsilon_m$量化的具体含义即为对每个时间步$t$模型$p_\theta$与真实环境$p$在策略$\pi_D$的状态分布下的TVD取最大值，因为误差可能随时间变化，而这是一种最坏情况的保守估计。

分布偏移误差$\epsilon_\pi$由所有状态下迭代前后策略之间的TVD最大值量化

$$
\epsilon_\pi=\max_sD_{TV}(\pi\|\pi_D)
$$

在工程实践中，泛化误差$\epsilon_m$可以通过验证损失来估计：先将$\pi_D$与真实环境$p$交互得到的数据分成训练集和验证集，在验证集上计算经训练集训练后的模型预测的损失，该损失近似于$\epsilon_m$。

而分布偏移误差$\epsilon_\pi$则通过**KL散度**（Kullback-Leibler Divergence）间接给出，因为TVD计算难度较大，而KL散度具有期望形式，可以利用蒙特卡洛估计

$$
D_{KL}(P\|Q)=\mathbb E_{x\sim P(x)}\left[\log\frac{P(x)}{Q(x)}\right]
$$

KL散度可以基于**Pinsker不等式**控制TVD的大小（$\log$以$2$为底时）

$$
\sqrt{\frac12D_{KL}(\pi\|\pi_D)}\geq D_{TV}^2(\pi\|\pi_D)
$$

MBPO证明了，性能差距上界$C$与泛化误差$\epsilon_m$和分布偏移误差$\epsilon_\pi$的关系为（证明见文末）

$$
C(\epsilon_m,\epsilon_\pi)=\frac{2\gamma r_{\max}(\epsilon_m+2\epsilon_\pi)}{(1-\gamma)^2}+\frac{4r_{\max}\epsilon_\pi}{1-\gamma}
$$

现在，我们只需将策略在模型中的期望累积回报$\hat\eta[\pi]$提高超过$C(\epsilon_m,\epsilon_\pi)$，就可以保证其在真实环境下期望累积回报的提高。而使用模型进行策略优化的算法设计关键就在于降低$C(\epsilon_m,\epsilon_\pi)$。

## 分支推演

单调改进的原始形式存在两个根本性缺陷：

- 当泛化误差$\epsilon_m$较高时，可能不存在使得$\hat\eta[\pi]$的提高超过$C(\epsilon_m,\epsilon_\pi)$的策略$\pi$；
- 对于较高的折扣因子$\gamma\rightarrow1$，因子$1/(1-\gamma)$会导致$C(\epsilon_m,\epsilon_\pi)$爆炸。其根本原因在于原始形式基于模型推演覆盖完整任务视野的假设，而更高的折扣因子需要更大的有效视界来准确估计回报，进而累积更多的误差（参见文末马尔可夫链TVD时变有界引理）。

在折扣因子作用下，智能体对未来奖励的重视程度随时间指数衰减。理论上累积回报的求和上限为无穷大，但其中大部分的贡献来自前若干步的奖励，而**有效视界**（Effective Horizon）则回答了主要贡献究竟前多少步奖励的问题。其中一个视角将有效视界定义为所有时间步按折扣系数的加权平均，即未来奖励的“平均发生时间”

$$
\mathbb E[T]=\frac{\displaystyle\sum^\infty_{t=0}t\cdot\gamma^t}{\displaystyle\sum^\infty_{t=0}\gamma^t}=\frac\gamma
{1-\gamma}
$$

当$\gamma\rightarrow1$时，$\mathbb E[T]\approx1/(1-\gamma)$。在原始的单调改进定理中，这意味着$C(\epsilon_m,\epsilon_\pi)$与有效视界的平方成正比。

最直接的解决方案是缩短模型的推演长度来减少模型累积误差，但在此之前，基于模型的策略优化算法（例如SLBO、PILCO等）都强制模型从初始状态开始推演，因此必须将推演长度与有效视界绑定才能生成全面的数据，且每轮只推演一条轨迹。为此MBPO提出了**分支推演**（Branched Rollout），模型推演的起点不再是初始状态$s_0$，而是可以从一条真实轨迹中任意时间步的状态$s_t$。由于推演起点对真实数据$s_t$的采样已经有效覆盖了任务全局，因此对推演长度也不再有要求，从而可以通过短推演减少模型累积误差。由此，MBPO将全局策略优化分解为了无数个局部策略改进。

![](https://inkem-1306784622.cos.accelerate.myqcloud.com/blog/pic/Pasted%20image%2020260209141407.png)

MBPO证明了，在步数为$k$的分支推演下，单调改进定理具有如下形式（证明见文末，但是博主没有理解论文的证明，进行了一系列缝缝补补，仅供参考）

$$
\eta[\pi] \geq \eta^{\text{branch}}[\pi] - 2r_{\max} \left[ \frac{\gamma^{k+1} \epsilon_\pi}{(1 - \gamma)^2} + \frac{\gamma^k + 2}{1 - \gamma} \epsilon_\pi + \frac{k}{1 - \gamma} (\epsilon_m + 2\epsilon_\pi) \right]
$$

其中$\eta^{\mathrm{branch}}[\pi]$为策略$\pi$在模型中执行$k$步推演的期望累积回报（$k$步之后的回报由价值函数给出）。

该形式表明，模型误差和分布偏移误差的影响正比于推演长度$k$，且分布偏移对误差有显著的放大作用，这将使线性项的增长淹没了指数项的衰减，因此降低性能差距上界的最优解是$k=0$，即不使用模型，这显然无法指导MBRL的算法设计。实际上，这一定理的证明过程悲观地承认了模型对所有分布偏移数据的误差，忽视了模型的泛化能力和平滑性质。论文通过实验证明了，分布偏移的误差放大作用会随着模型学习到的数据量的增长而减小。

![](https://inkem-1306784622.cos.accelerate.myqcloud.com/blog/pic/Pasted%20image%2020260209144017.png)

_随着模型学习的数据量增加，策略分布偏移带来的整体误差更低，且误差随着策略分布偏移的增长更平缓。_

为此，MBPO对性能差距上界进一步修正，将线性项$\epsilon_{m'}=\epsilon_m + 2\epsilon_\pi$修改为一阶线性近似

$$
\hat\epsilon_{m'}(\epsilon_\pi)\approx\epsilon_m+\epsilon_\pi\cdot\frac{\mathrm d\epsilon_{m'}}{\mathrm d\epsilon_\pi}
$$

只要模型学习到足够的数据，一阶线性项的增长将不再具有主导作用，而是可以在$k$较小的一个区间内被指数项的衰减抵消，使得模型分支推演的有效性具有合理性，从而充分利用MBRL高样本效率的优势。

通过$k$的取值，分支推演统一了基于模型方法和无模型方法：

- $k=0$：无模型方法，单调改进定理退化为外推误差问题，即新旧策略分布的差异导致由旧策略数据训练的价值函数估计不可靠；
- $k=1$：单步推演，与Dyna架构一致；
- $k=1\sim5$：短分支推演，MBPO的推荐范围；
- $k=\infty$：全长推演，SLBO、PILCO等方法。

在训练初期，模型质量较差，可以优先采取$k=0$或$1$。随着模型得到有效的学习，可以逐渐增大$k$的取值以提高样本效率。这正对应了论文的标题——“何时信任你的模型”（When to Trust Your Model）。

---

MBPO采用了SAC作为策略优化算法和PETS中的概率集成模型，其算法流程如下

1. 初始化策略$\pi_\phi$，预测模型$p_\theta$，环境数据集$\mathcal D_{\mathrm{env}}$和模型数据集$\mathcal D_{\mathrm{model}}$；
2. 对于$N$个训练轮次执行：
   1. 通过最大似然估计在$\mathcal D_{\mathrm{env}}$上训练模型$p_\theta$；
   2. 对于$E$步环境交互执行：
      1. 根据策略$\pi_\phi$在真实环境中行动，收集数据到$\mathcal D_{\mathrm{env}}$；
      2. 对于$M$次模型推演执行：
         1. 从$\mathcal D_{\mathrm{env}}$中均匀采样一个状态$s_t$；
         2. 从$s_t$出发，使用策略$\pi_\phi$进行$k$步模型推演，收集数据到$\mathcal D_{\mathrm{model}}$。
      3. 对于$G$次梯度更新执行：
         1. 通过策略优化算法在$\mathcal D_{\mathrm{model}}$上更新策略参数（小批量样本）：$$\phi\leftarrow\phi-\lambda_\pi\hat\nabla_\phi J_\pi(\phi,\mathcal D_{\mathrm{model}})$$

为了保证策略既能充分利用当前模型生成的数据，又不在当前模型上过拟合，$G$的取值应适中，MBPO推荐值为$20$。模型数据集$\mathcal D_{\mathrm{model}}$仅基于当前策略生成，将在每个策略优化阶段结束后清空。

## 基于完整推演的单调改进定理证明

完整推演的单调改进定理形式如下：假设两个模型在每个时间步上的期望TVD有界

$$
\epsilon_m=\max_t\mathbb E_{s\sim\pi_{D,t}}[D_{TV}(p(s',r|s,a)\|p_\theta(s',r|s,a))]
$$

且生成模型训练数据的策略$\pi_D$与新策略$\pi$的TVD有界

$$
\epsilon_\pi=\max_sD_{TV}(\pi\|\pi_D)
$$

则有

$$
\eta[\pi]\geq\hat\eta[\pi]-\frac{2\gamma r_{\max}(\epsilon_m+2\epsilon_\pi)}{(1-\gamma)^2}-\frac{4r_{\max}\epsilon_\pi}{1-\gamma}
$$

其关键在于求解模型与真实环境性能差距$\eta[\pi]-\hat\eta[\pi]$的最大值。假设当前模型由$\pi_D$收集的训练数据训练，我们对其做如下变换和误差分解

$$
\eta[\pi]-\hat\eta[\pi]=\underset{L_1}{\underbrace{\eta[\pi]-\eta[\pi_D]}}+\underset{L_2}{\underbrace{\eta[\pi_D]-\hat\eta[\pi_D]}}+\underset{L_3}{\underbrace{\hat\eta[\pi_D]-\hat\eta[\pi]}}
$$

现在我们将性能差距分解为了仅涉及策略差异的$L_1$、$L_3$和仅涉及模型误差的$L_2$。

_论文此处没有将$L_2$和$L_3$分解，博主认为这样更好理解。_

---

**回报差异有界**（Branched Returns Bound）是证明单调改进定理的核心引理：假设两个动力学分布之间的期望KL散度有界

$$
\max_t\mathbb E_{s\sim p^t_1(s)}D_{KL}(p_1(s',a|s)\|p_2(s',a|s))\leq\epsilon_m
$$

且策略差异有界

$$
\max_sD_{TV}(\pi_1(a|s)\|\pi_2(a|s))\leq\epsilon_\pi
$$

则回报差异有界

$$
|\eta_1-\eta_2|\leq\frac{2r_{\max}\gamma(\epsilon_\pi+\epsilon_m)}{(1-\gamma)^2}+\frac{2r_{\max}\epsilon_\pi}{(1-\gamma)}
$$

其中$\eta_1$和$\eta_2$分别表示$\pi_1$在动力学分布$p_1(s'|s,a)$下的期望回报和$\pi_2$在动力学分布$p_2(s'|s,a)$下的期望回报。

其证明如下，首先将$|\eta_1-\eta_2|$展开

$$
\begin{split}
|\eta_1-\eta_2|&=\left|\sum_{s,a}\sum_t\gamma^t(p_1^t(s,a)-p_2^t(s,a))r(s,a)\right|\\
&\leq\sum_t\sum_{s,a}\gamma^t|p_1^t(s,a)-p_2^t(s,a)|r(s,a)\\
&\leq r_{\max}\sum_t\sum_{s,a}\gamma^t|p_1^t(s,a)-p_2^t(s,a)|\\
&=2r_{\max}\sum_t\gamma^tD_{TV}(p_1^t(s,a)\|p_2^t(s,a))
\end{split}
$$

---

针对$D_{TV}(p_1^t(s,a)\|p_2^t(s,a))$一项，MBPO提出了**联合分布TVD有界**引理，将状态-动作对分布TVD分解为状态分布TVD和策略分布TVD。即一对联合分布$p_1(x,y)=p_1(x)p_1(y|x)$和$p_2(x,y)=p_2(x)p_2(y|x)$的TVD有界，且其上界可以由构成联合分布的边际分布和条件分布的TVD表示

$$
D_{TV}(p_1(x,y)\|p_2(x,y))\leq D_{TV}(p_1(x)\|p_2(x))+\max_xD_{TV}(p_1(y|x)\|p_2(y|x))
$$

或者采用更紧的上界

$$
D_{TV}(p_1(x,y)\|p_2(x,y))\leq\mathbb E_{x\sim p_1}[D_{TV}(p_1(y|x)\|p_2(y|x))]+D_{TV}(p_1(x)\|p_2(x))
$$

其证明如下

$$
\begin{split}
D_{TV}(p_1(x,y)\|p_2(x,y))&=\frac12\sum_{x,y}|p_1(x,y)-p_2(x,y)|\\
&=\frac12\sum_{x,y}|p_1(x)p_1(y|x)-p_2(x)p_2(y|x)|\\
&=\frac12\sum_{x,y}|p_1(x)p_1(y|x)-p_1(x)p_2(y|x)+p_1(x)p_2(y|x)-p_2(x)p_2(y|x)|\\
&\leq\frac12\sum_{x,y}\Big(p_1(x)|p_1(y|x)-p_2(y|x)|+|p_1(x)-p_2(x)|p_2(y|x)\Big)\\
&=\sum_{x}p_1(x)\sum_y\frac12|p_1(y|x)-p_2(y|x)|+\frac12\sum_{x}\underset1{\underbrace{\sum_yp_2(y|x)}}|p_1(x)-p_2(x)|\\
&=\mathbb E_{x\sim p_1}[D_{TV}(p_1(y|x)\|p_2(y|x))]+D_{TV}(p_1(x)\|p_2(x))\\
&\leq\max_xD_{TV}(p_1(y|x)\|p_2(y|x))+D_{TV}(p_1(x)\|p_2(x))
\end{split}
$$

将其应用于$D_{TV}(p_1^t(s,a)\|p_2^t(s,a))$，我们有

$$
D_{TV}(p_1^t(s,a)\|p_2^t(s,a))\leq\max_sD_{TV}(\pi_1(a|s)\|\pi_2(a|s))+D_{TV}(p_1(s)\|p_2(s))
$$

---

$\max_sD_{TV}(\pi_1(a|s)\|\pi_2(a|s))$正是分布偏移误差$\epsilon_\pi$。而针对$D_{TV}(p_1(s)\|p_2(s))$一项，MBPO提出了**马尔可夫链TVD时变有界**引理：假设两个马尔可夫链在每个时间步的状态转移分布（动力学分布和策略分布的联合分布）之间的最大期望KL散度有界

$$
\max_t\mathbb E_{s\sim p^t_1(s)}D_{KL}(p_1(s'|s)\|p_2(s'|s))\leq\delta
$$

且初始状态分布相同

$$
p_1^{t=0}(s)=p_2^{t=0}(s)
$$

则每个时间步二者所处状态的边际分布的TVD（记为$\epsilon_t$）也有界，且这个上界随着步数$t$线性增长

$$
\epsilon_t=D_{TV}(p_1^t(s)\|p_2^t(s))\leq t\delta
$$

其证明如下：首先展开$p_1^t$和$p_2^t$的L1（绝对值）距离，当前时间步的状态分布由上一时间步的状态分布和状态转移分布共同决定（此处$s'$是上一时间步的状态而非后继状态）

$$
\begin{split}
|p_1^t(s)-p_2^t(s)|&=\left|\sum_{s'}\Big(p_1(s_t=s|s')p_1^{t-1}(s')-p_2(s_t=s|s')p_2^{t-1}(s')\Big)\right|\\
&\leq\sum_{s'}|p_1(s_t=s|s')p_1^{t-1}(s')-p_2(s_t=s|s')p_2^{t-1}(s')|\\
&=\sum_{s'}|p_1(s|s')p_1^{t-1}(s')-p_2(s|s')p_1^{t-1}(s')+p_2(s|s')p_1^{t-1}(s')-p_2(s|s')p_2^{t-1}(s')|\\
&\leq\sum_{s'}\Big(p_1^{t-1}(s')|p_1(s|s')-p_2(s|s')|+p_2(s|s')|p_1^{t-1}(s')-p_2^{t-1}(s')|\Big)\\
&=\mathbb E_{s'\sim p_1^{t-1}}[|p_1(s|s')-p_2(s|s')|]+\sum_{s'}p_2(s|s')|p_1^{t-1}(s')-p_2^{t-1}(s')|
\end{split}
$$

现在将L1距离展开代入计算$p_1^t$和$p_2^t$的TVD

$$
\begin{split}
\epsilon_t&=\frac12\sum_s|p_1^t(s)-p_2^t(s)|\\
&=\frac12\sum_s\Big(\mathbb E_{s'\sim p_1^{t-1}}[|p_1(s|s')-p_2(s|s')|]+\sum_{s'}p_2(s|s')|p_1^{t-1}(s')-p_2^{t-1}(s')|\Big)\\
&=\mathbb E_{s'\sim p^{t-1}_1}\left[\frac12\sum_s|p_1(s|s')-p_2(s|s')|\right]+\frac12\sum_{s'}\sum_sp_2(s|s')|p_1^{t-1}(s')-p_2^{t-1}(s')|\\
\end{split}
$$

右侧第一项即为状态转移分布在$t-1$时的期望KL散度，记为$\delta_{t-1}$，则其上界为$\delta$。第二项中$\sum_sp_2(s|s')=1$，约去后即为$p_1^{t-1}$和$p_2^{t-1}$的TVD，形成递归。现在有

_论文此处下标有误，博主做出了修改。_

$$
\begin{split}
\epsilon_t&=\delta_{t-1}+\epsilon_{t-1}\\
&=\epsilon_0+\sum^{t-1}_{i=0}\delta_i\\
&\leq t\delta\\
\end{split}
$$

我们假设两个马尔可夫链采用相同的初始状态分布，即$\epsilon_0=0$。

$\delta$可以利用联合分布TVD有界引理给出。任意时间步$t$下的状态转移分布有

$$
\begin{split}
D_{KL}(p_1^t(s'|s)\|p_2^t(s'|s))&\leq\max_sD_{TV}(\pi_1(a|s)\|\pi_2(a|s))+D_{TV}(p_1(s'|s,a)\|p_2(s'|s,a))\\
&=\epsilon_\pi+\epsilon_m
\end{split}
$$

则

$$
\begin{split}
\max_t\mathbb E_{s\sim p^t_1(s)}D_{KL}(p_1(s'|s)\|p_2(s'|s))&\leq\max_{t,s,s'}D_{KL}(p_1^t(s'|s)\|p_2^t(s'|s))\\
&\leq\epsilon_\pi+\epsilon_m=\delta
\end{split}
$$

---

回到回报差异有界引理的证明，现在对$D_{TV}(p_1^t(s,a)\|p_2^t(s,a))$有

$$
\begin{split}
D_{TV}(p_1^t(s,a)\|p_2^t(s,a))&\leq\max_sD_{TV}(\pi_1(a|s)\|\pi_2(a|s))+D_{TV}(p_1(s)\|p_2(s))\\
&\leq\epsilon_\pi+t(\epsilon_\pi+\epsilon_m)
\end{split}
$$

则

$$
\begin{split}
|\eta_1-\eta_2|&=2r_{\max}\sum_t\gamma^tD_{TV}(p_1^t(s,a)\|p_2^t(s,a))\\
&\leq2r_{\max}\sum_t\gamma^t(t(\epsilon_\pi+\epsilon_m)+\epsilon_\pi)\\
&\leq2r_{\max}\left(\frac{\gamma(\epsilon_\pi+\epsilon_m)}{(1-\gamma)^2}+\frac{\epsilon_m}{1-\gamma}\right)
\end{split}
$$

---

回到单调改进定理

$$
\eta[\pi]-\hat\eta[\pi]=\underset{L_1}{\underbrace{\eta[\pi]-\eta[\pi_D]}}+\underset{L_2}{\underbrace{\eta[\pi_D]-\hat\eta[\pi_D]}}+\underset{L_3}{\underbrace{\hat\eta[\pi_D]-\hat\eta[\pi]}}
$$

应用回报差异有界引理，对于$L_1$和$L_3$有$\epsilon_m=0$，对于$L_2$有$\epsilon_\pi=0$，则

$$
\begin{split}
\eta[\pi]-\hat\eta[\pi]&\leq2\cdot2r_{\max}\frac{\gamma\epsilon_\pi}{(1-\gamma)^2}+2r_{\max}\left(\frac{\gamma\epsilon_m}{(1-\gamma)^2}+\frac{\epsilon_m}{1-\gamma}\right)\\
&=\frac{2\gamma r_{\max}(\epsilon_m+2\epsilon_\pi)}{(1-\gamma)^2}+\frac{4r_{\max}\epsilon_\pi}{1-\gamma}
\end{split}
$$

我们不能直接对$\eta[\pi]-\hat\eta[\pi]$应用回报差异有界引理，因为模型由$\pi_D$收集的数据训练，只能保证模型误差在$\pi_D$的状态分布上是有界的，但在新策略$\pi$的状态分布上则不一定。

## 基于分支推演的单调改进定理证明（参考）

分支推演的单调改进定理形式如下：假设两个模型在每个时间步上的期望TVD有界

$$
\epsilon_m=\max_t\mathbb E_{s\sim\pi_{D,t}}[D_{TV}(p(s',r|s,a)\|p_\theta(s',r|s,a))]
$$

且生成模型训练数据的策略$\pi_D$与新策略$\pi$的TVD有界

$$
\epsilon_\pi=\max_sD_{TV}(\pi\|\pi_D)
$$

则有

$$
\eta[\pi] \geq \eta^{\text{branch}}[\pi]-2r_{\max}\left[ \frac{\gamma^{k+1} \epsilon_\pi}{(1 - \gamma)^2} + \frac{\gamma^k + 2}{1 - \gamma} \epsilon_\pi + \frac{k}{1 - \gamma} (\epsilon_m + 2\epsilon_\pi) \right]
$$

其证明如下，我们记智能体从分支起点开始，在真实环境下执行$k$步旧策略$\pi_D$，之后执行新策略$\pi$的期望累积回报为$\eta^{\pi_D,\pi}$。$\eta[\pi]-\eta^{\text{branch}}[\pi]$可做如下变换和误差分解

$$
\eta[\pi]-\eta^{\text{branch}}[\pi]=\underset{L_1}{\underbrace{\eta[\pi]-\eta^{\pi_D,\pi}}}+\underset{L_2}{\underbrace{\eta^{\pi_D,\pi}-\eta^{\mathrm{branch}}}}
$$

其中$L_1$仅包含$k$步内的策略差异，而$L_2$既有模型误差又有策略差异。

---

MBPO首先进一步给出回报差异边界在分支推演下的形式：假设分支推演长度为$k$，在从推演起点状态开始的$k$个时间步内（pre分支）和分支推演结束后的时间步（post分支），两个动力学分布的KL散度和策略差异均有界

$$
\begin{split}
\max_t\mathbb E_{s\sim p^t_1(s)}D_{KL}(p_1^{\mathrm{pre}}(s',a|s)\|p_2^{\mathrm{pre}}(s',a|s))\leq\epsilon_m^{\mathrm{pre}}\\
\max_t\mathbb E_{s\sim p^t_1(s)}D_{KL}(p_1^{\mathrm{post}}(s',a|s)\|p_2^{\mathrm{post}}(s',a|s))\leq\epsilon_m^{\mathrm{post}}\\
\max_sD_{TV}(\pi_1^{\mathrm{pre}}(a|s)\|\pi_2^{\mathrm{pre}}(a|s))\leq\epsilon_\pi^{\mathrm{pre}}\\
\max_sD_{TV}(\pi_1^{\mathrm{post}}(a|s)\|\pi_2^{\mathrm{post}}(a|s))\leq\epsilon_\pi^{\mathrm{post}}
\end{split}
$$

_由于模型只推演$k$步，因此在实际算法中只有pre分支的模型分布是显式存在的，而post分支的模型分布则由价值函数经模型数据训练后隐式建模。_

则回报差异有界

$$
|\eta_1-\eta_2|\leq2r_{\max}\left[\frac{\gamma^{k+1}}{(1-\gamma)^2}(\epsilon_m^{\mathrm{post}}+\epsilon_\pi^{\mathrm{post}})+\frac k{1-\gamma}(\epsilon_m^{\mathrm{pre}}+\epsilon_\pi^{\mathrm{pre}})+\frac{\gamma^{k+1}}{1-\gamma}\epsilon_\pi^{\mathrm{post}}+\frac1{1-\gamma}\epsilon_\pi^{\mathrm{pre}}\right]
$$

_此处论文下标pre和post疑似写反，同时“分支前”和“分支后”的描述存在歧义，博主怀疑为“分支期间”和“分支结束后”。_

其证明如下，与原始的回报差异有界引理类似，我们可以先将$|\eta_1-\eta_2|$推导至如下不等式

$$
\begin{split}
|\eta_1-\eta_2|&\leq2r_{\max}\sum_t\gamma^tD_{TV}(p_1^t(s,a)\|p_2^t(s,a))
\end{split}
$$

_此处论文转而用和原始回报差异有界引理不一样的基于占用度量的推导方法，虽然结果等价，但令博主感到莫名其妙所以没有采纳。_

根据马尔可夫链TVD时变有界引理，$D_{TV}(p_1^t(s,a)\|p_2^t(s,a))$将如下分阶段累积误差

$$
D_{TV}(p_1^t(s,a)\|p_2^t(s,a))\leq\left\{\begin{matrix}t(\epsilon_m^{\mathrm{pre}}+\epsilon_\pi^{\mathrm{pre}})+\epsilon_\pi^{\mathrm{pre}}\leq k(\epsilon_m^{\mathrm{pre}}+\epsilon_\pi^{\mathrm{pre}})+\epsilon_\pi^{\mathrm{pre}},t\leq k\\(t-k)(\epsilon_m^{\mathrm{post}}+\epsilon_\pi^{\mathrm{post}})+k(\epsilon_m^{\mathrm{pre}}+\epsilon_\pi^{\mathrm{pre}})+\epsilon_\pi^{\mathrm{post}}+\epsilon_\pi^{\mathrm{pre}},t\geq k\end{matrix}\right.
$$

将其在所有时间步上按折扣系数加权求和

$$
\begin{split}
&\sum_{t=0}^\infty\gamma^tD_{TV}(p_1^t(s,a)\|p_2^t(s,a))\\
=&\sum_{t=0}^k\gamma^t(k(\epsilon_m^{\mathrm{pre}}+\epsilon_\pi^{\mathrm{pre}})+\epsilon_\pi^{\mathrm{pre}})+\sum_{t=k+1}^\infty\gamma^t((t-k)(\epsilon_m^{\mathrm{post}}+\epsilon_\pi^{\mathrm{post}})+k(\epsilon_m^{\mathrm{pre}}+\epsilon_\pi^{\mathrm{pre}})+\epsilon_\pi^{\mathrm{post}}+\epsilon_\pi^{\mathrm{pre}})\\
=&\frac{\gamma^{k+1}}{(1-\gamma)^2}(\epsilon_m^{\mathrm{post}}+\epsilon_\pi^{\mathrm{post}})+\frac k{1-\gamma}(\epsilon_m^{\mathrm{pre}}+\epsilon_\pi^{\mathrm{pre}})+\frac{\gamma^{k+1}}{1-\gamma}\epsilon_\pi^{\mathrm{post}}+\frac1{1-\gamma}\epsilon_\pi^{\mathrm{pre}}
\end{split}
$$

将其乘以$2r_{\max}$即可得到回报差异上界。

_此处论文分阶段求和下标对第$k$项重复求和，且第三项分子为$\gamma^k$，博主无法理解。_

---

回到单调改进定理

$$
\eta[\pi]-\eta^{\text{branch}}[\pi]=\underset{L_1}{\underbrace{\eta[\pi]-\eta^{\pi_D,\pi}}}+\underset{L_2}{\underbrace{\eta^{\pi_D,\pi}-\eta^{\mathrm{branch}}}}
$$

$L_1$只包含分支期间的策略差异，故取$\epsilon_\pi^{\mathrm{pre}}\leq\epsilon_\pi$，其余误差为$0$，有

$$
|L_1|\leq2r_{\max}\frac{k+1}{1-\gamma}\epsilon_\pi
$$

$L_2$需继续进行误差分解如下

$$
L_2=\underset{L_3}{\underbrace{\eta^{\pi_D,\pi}-\eta^{\pi_D,\hat\pi_D}}}+\underset{L_4}{\underbrace{\eta^{\pi_D,\hat\pi_D}-\eta^{\pi,\hat\pi_D}}}+\underset{L_5}{\underbrace{\eta^{\pi,\hat\pi_D}-\eta^{\pi,\hat\pi}}}+\underset{L_6}{\underbrace{\eta^{\pi,\hat\pi}-\eta^{\mathrm{branch}}}}
$$

其中$\eta^{\pi_D,\hat\pi_D}$表示从分支起点开始，在真实环境下执行$k$步旧策略$\pi_D$，之后在模型中执行旧策略$\pi_D$的期望累积回报，其他以此类推。则$L_3$包含分支结束后的模型误差和策略差异，$L_4$包含分支期间策略差异，$L_5$包含分支后策略差异，$L_6$包含分支期间模型误差。

_此处论文仅展开到$L_3$，后续部分声称仅包含post分支策略差异，博主无法理解。_

$$
\begin{split}
\frac{|L_2|}{2r_{\max}}&\leq\left(\frac{\gamma^{k+1}}{(1-\gamma)^2}(\epsilon_m+\epsilon_\pi)+\frac{\gamma^{k+1}}{1-\gamma}\epsilon_\pi\right)+\left(\frac {k+1}{1-\gamma}\epsilon_\pi\right)+\left(
\frac{\gamma^{k+1}}{(1-\gamma)^2}\epsilon_\pi+\frac{\gamma^{k+1}}{1-\gamma}\epsilon_\pi\right)+\left(\frac k{1-\gamma}\epsilon_m\right)\\
\end{split}
$$

最终代回原式即可。

得出这个结果博主已经力竭了。虽然与论文有所出入，但不影响分支推演步数$k$在性能差距上界中的地位，尤其是线性项，因此基于单调改进定理得出的算法设计原则依旧可靠。

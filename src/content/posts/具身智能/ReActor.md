---
title: 【ReActor】面向物理感知运动重定向的强化学习
date: 2026-05-30
summary: 重定向和跟踪不是两个独立的问题，而是一个协同优化的整体。
tags: [ReActor, 运动重定向, 强化学习, 人形机器人, 论文精读]
category: 具身智能
---

::bilibili{#BV1bXRmB9EN1}

- 论文标题：_ReActor: Reinforcement Learning for Physics-Aware Motion Retargeting_
- 发表时间：2026.05.07
- 论文链接：https://arxiv.org/abs/2605.06593
- 补充知识：运动重定向

# 一、运动重定向问题

**运动重定向**（Motion Retargeting）是动画、游戏和机器人领域的核心问题之一，用于解决如何将已有的运动数据（如人类动作捕捉）迁移到不同形态的角色或机器人上，同时保留原始运动的语义和风格特征，从而高效、充分地利用数据。对于动画和游戏中的虚拟角色而言，运动重定向技术主要追求视觉上的形似和自然。然而在机器人领域，运动重定向技术不仅面临人类与机器人在骨骼结构、肢体比例、关节自由度等方面的显著差异，还要解决物理世界中的可行性问题，直接复制关节角度会导致完全不可用的结果。

物理不可行是运动重定向长期存在的难题，又称**伪影**（Artifacts）。最典型的伪影问题包括：

- 脚底滑动：角色脚底本应踩实地面，却在空中滑动或像溜冰一样平移。
- 自穿透：角色的肢体穿过了自身身体。
- 地面穿透：角色的肢体穿入了地面之下。
- 关节伸展过度：角色的关节运动超越了生理或物理极限。

![](https://inkem-1306784622.cos.accelerate.myqcloud.com/blog/pic/Pasted%20image%2020260528191932.png)

运动重定向技术的演变主要经历了三个阶段：

- 第一阶段：基于传统运动学，通过优化算法最小化源角色与目标角色之间的姿态差异。
- 第二阶段：引入物理约束（如力矩限制、接触力），确保重定向后的运动在物理上可行。
- 第三阶段：利用深度学习或强化学习自动学习源到目标的映射，减少人工干预。

现有的学习式重定向方法普遍采用先翻译再模仿的双阶段串行架构：

- 第一阶段：使用逆运动学优化算法，将人类动作独立翻译为机器人的关节角度序列，目标是让机器人的姿态在几何上接近人类。
- 第二阶段：训练一个强化学习策略，目标是在跟踪第一阶段得到的参考轨迹的同时保证物理可行性。

然而该架构存在的缺陷是，第一阶段需要针对不同的机器人型号和运动数据来源进行大量手工调参。并且强化学习的纠正能力有限，如果第一阶段翻译出来的动作具有显著的伪影问题，强化学习将难以兼顾几何跟踪效果和物理可行性。

为此，ReActor选择打破翻译和模仿之间的壁垒，采用**双层优化**（Bilevel Optimization）框架。上层负责优化重定向参数，下层负责强化学习训练，二者协同调整。这意味着如果下层策略在跟踪中遇到困难，上层优化器会自动调整重定向参数以适应机器人的物理能力。

# 二、双层优化框架

双层优化框架定义了一个嵌套式优化问题

$$
\min_{\mathbf p\in\mathcal P}\mathcal L(\mathbf p,\phi^*(\mathbf p))\quad\text{subject to}\quad\phi^*(\mathbf p)=\arg\max\mathcal R(\mathbf p,\phi)
$$

- 上层问题：给定下层最优策略$\phi^*$，优化重定向参数$\mathbf p$以最小化损失函数$\mathcal L$。
- 下层问题：给定上层重定向参数$\mathbf p$，优化策略$\phi$以最大化奖励函数$\mathcal R$。

具体而言，上层通过重定向参数$\mathbf p$，将源参考运动$\mathbf m_t$映射为目标参考运动$\mathbf g_t$，用户只需手动为源角色与目标角色选择需要匹配的刚体；下层通过强化学习训练得到最优策略$\phi^*$，并利用最优策略推演轨迹序列$\mathbf s^*_t$，其相当于机器人学到的实际运动。

损失函数$\mathcal L$即为给定最优策略$\phi^*$、轨迹初始状态$\mathbf s_0$和源参考运动$\mathbf m_t$时，轨迹序列$\mathbf s^*_t$与目标参考运动$\mathbf g_t$的期望误差

$$
\mathcal L(\mathbf p,\phi^*(\mathbf p))=\mathbb E_{\pi_{\phi^*},\mathbf s_0,\mathbf m_t}[\ell(\mathbf g_t-\mathbf s^*_t)]
$$

![](https://inkem-1306784622.cos.accelerate.myqcloud.com/blog/pic/Pasted%20image%2020260529192854.png)

标准的双层优化流程是先解决下层的优化问题，再解决上层的优化问题。但是每次更新上层参数前都等待下层强化学习收敛会耗费大量时间，为此ReActor采用了基于**双时间尺度近似**（Two-Timescale Approximation，TTSA）的**单循环**（Single Loop）双层优化算法。简而言之，上层参数可以在强化学习每轮迭代之后都进行调整，但是采用更小的学习率使其更新速度稍慢，避免系统震荡。

上层参数的更新公式为

$$
\mathbf p\leftarrow P_\mathcal P(\mathbf p-\eta\tilde{\mathrm d}_\mathbf p\mathcal L)
$$

- $\tilde{\mathrm d}_\mathbf p\mathcal L$：上层损失$\mathcal L$关于重定向参数$\mathbf p$的梯度估计。
- $\eta$：学习率。
- $P_\mathcal P(\cdot)$：投影操作。

上层优化问题的核心便在于计算梯度估计。对于损失函数期望内部的误差函数，我们可以展开其梯度

$$
\mathrm d_\mathbf p\ell=\partial_{\mathbf g_t}\ell\mathrm d_\mathbf p\mathbf g_t+\partial_{\mathbf s^*_t}\ell\mathrm d_\mathbf p\mathbf s^*_t
$$

该式的含义是调整参数$\mathbf p$对跟踪误差$\ell$的影响：

- 第一项$\partial_{\mathbf g_t}\ell\mathrm d_\mathbf p\mathbf g_t$表示参数$\mathbf p$通过改变参考运动$\mathbf g_t$对误差$\ell$的影响，由于参考运动$\mathbf g_t$与参数$\mathbf p$的数学关系是确定的（见后文），所以这一项可以直接计算。
- 第二项$\partial_{\mathbf s^*_t}\ell\mathrm d_\mathbf p\mathbf s^*_t$表示参数$\mathbf p$通过改变参考运动$\mathbf g_t$，进而导致强化学习策略的表现$\mathbf s^*_t$改变对误差$\ell$的影响。但是直接计算这一项需要展开强化学习策略这个“黑箱”，在数学上极其困难。

为了简化第二项，ReActor提出了两个关键假设：

**假设一**：误差函数$\ell$只关心参考运动与实际运动之间的差值，如$\ell=(\mathbf g_t-\mathbf s_t^*)^2$。

在该假设下，机器人实际位置$\mathbf s_t^*$远离参考位置$\mathbf g_t$一段距离$\Delta$，和参考位置$\mathbf g_t$远离机器人实际位置$\mathbf s_t^*$一段距离$\Delta$，对误差影响的大小是一样的，只是方向相反，即

$$
\partial_{\mathbf s^*_t}\ell=-\partial_{\mathbf g_t}\ell
$$

**假设二**：当参考运动改变时，强化学习策略不会完全跟随其改变，也不会完全不变，而是跟随到一部分改变。

具体而言，如果参考运动$\mathbf g_t$改变了$\Delta$，则实际运动$\mathbf s_t^*$会改变$\alpha\Delta$，其中$\alpha\in[0,1]$。ReActor用对角矩阵$\alpha I$来建模这一部分跟随行为，单位矩阵$I$意味着强化学习策略对参考运动的改变在各个方向上的跟随程度都是一样的。在该假设下，我们有

$$
\mathrm d_\mathbf p\mathbf s^*_t=\alpha\mathrm d_\mathbf p\mathbf g_t
$$

根据上述假设，误差函数的梯度可简化为如下估计

$$
\begin{split}
\tilde{\mathrm d}_\mathbf p\ell&=\partial_{\mathbf g_t}\ell\mathrm d_\mathbf p\mathbf g_t+\partial_{\mathbf s^*_t}\ell\mathrm d_\mathbf p\mathbf s^*_t\\
&=\partial_{\mathbf g_t}\ell\mathrm d_\mathbf p\mathbf g_t+(-\partial_{\mathbf g_t}\ell)(\alpha\mathrm d_\mathbf p\mathbf g_t)\\
&=(1-\alpha)\partial_{\mathbf g_t}\ell\mathrm d_\mathbf p\mathbf g_t
\end{split}
$$

该梯度估计可以被直接计算。

最终给定一批数据$\mathcal D$，损失函数的梯度估计为

$$
\tilde{\mathrm d}_{\mathrm p}\mathcal L=\frac1{|\mathcal D|}\sum_{(\mathbf s_t,\mathbf g_t)\in\mathcal D}\tilde{\mathrm d}_{\mathbf p}\ell(\mathbf g_t-\mathbf s_t)
$$

# 三、重定向参数化

重定向参数化定义了如何通过一组可优化的参数，将源角色的运动数据转换成目标机器人的参考运动。

![](https://inkem-1306784622.cos.accelerate.myqcloud.com/blog/pic/Pasted%20image%2020260529192700.png)

## 1. 用户输入

用户需要将源角色与目标机器人设置为同一标准姿态，通常使用最容易建立空间对应关系的T\-pose（双臂平伸站立）。随后指定源角色与目标机器人之间语义上对应的刚体对，对应关系可以是稀疏的，即不要求所有的刚体都得到匹配。

现在，对于每个刚体对$b$，我们有源刚体的位置$\mathbf x_\mathrm{source}^b$和姿态$\mathbf R_\mathrm{source}^b$，以及目标刚体的位置$\mathbf x_\mathrm{target}^b$和姿态$\mathbf R_\mathrm{target}^b$。

## 2. 全局缩放与名义变换

全局缩放根据源角色与目标机器人的体型差异，将源刚体的位置整体缩放到目标刚体的尺度。全局缩放因子$s$是目标机器人与源角色的高度之比

$$
\textcolor{cyan}s=\frac{h_\mathrm{target}}{h_\mathrm{source}}
$$

名义变换用于计算在源刚体坐标系下，源刚体到目标刚体的位置偏移$\mathbf x_\mathrm{nom}^b$与朝向旋转$\mathbf R_\mathrm{nom}^b$

$$
\begin{split}
\textcolor{cyan}{\mathbf x_\mathrm{nom}^b}&=(\mathbf R_\mathrm{source}^b)^T(\mathbf x_\mathrm{target}^b-\textcolor{cyan}s\mathbf x_\mathrm{source}^b)\\
\textcolor{cyan}{\mathbf R_\mathrm{nom}^b}&=(\mathbf R_\mathrm{source}^b)^T\mathbf R_\mathrm{target}^b
\end{split}
$$

名义变换是在标准姿态下计算的固定的映射规则，它补偿了源角色和目标机器人坐标系定义规则的差异，以及形态上以标准姿态为依据的静态基准差异。名义变换将会作为一个固有变换被应用到运动的每一帧中，因此由源刚体坐标系描述，后续再转换为世界坐标系描述。

## 3. 参数化变换与垂直偏移

现在，给定源参考动作$\mathbf m_t$，在名义变换的基础上引入重定向参数，即可生成最终的目标参考动作$\mathbf g_t$。重定向的运动量不仅包含位置$\mathbf x$和姿态$\mathbf R$，还有线速度$\mathbf v$和角速度$\boldsymbol\omega$，进一步提供动作的节奏与风格信息。源参考动作和目标参考动作的运动量均在世界坐标系下描述，由如下四个公式转换

$$
\begin{split}
\mathbf x^b_{\mathbf g_t}&=\mathbf R^b_{\mathbf m_t}(\textcolor{cyan}{\mathbf R^b_\mathrm{nom}}\textcolor{pink}{\mathbf p^b_\mathrm{pos}}+\textcolor{cyan}{\mathbf x^b_\mathrm{nom}})+\textcolor{cyan}s\mathbf x^b_{\mathbf m_t}+(\textcolor{cyan}{z_\mathrm{nom}}+\textcolor{pink}{p_z})\mathbf e_z\\
\mathbf R^b_{\mathbf g_t}&=\mathbf R^b_{\mathbf m_t}\textcolor{cyan}{\mathbf R^b_\mathrm{nom}}\mathrm{Exp}(\textcolor{pink}{\mathbf p^b_\mathrm{ori}})\\
\mathbf v^b_{\mathbf g_t}&=\boldsymbol\omega^b_{\mathbf m_t}\times\mathbf R^b_{\mathbf m_t}(\textcolor{cyan}{\mathbf R^b_\mathrm{nom}}\textcolor{pink}{\mathbf p^b_\mathrm{pos}}+\textcolor{cyan}{\mathbf x^b_\mathrm{nom}})+\textcolor{cyan}s\mathbf v^b_{\mathbf m_t}\\
\boldsymbol\omega^b_{\mathbf g_t}&=\boldsymbol\omega^b_{\mathbf m_t}
\end{split}
$$

其中

- 位置调整参数$\mathbf p^b_{\mathrm{pos}}$：三维位置向量，先通过$\mathbf R_{\mathrm{nom}}^b$旋转到描述名义位置偏移$\mathbf x_{\mathrm{nom}}^b$的坐标系下，再与之相加。
- 旋转调整参数$\mathbf p^b_{\mathrm{ori}}$：三维旋转向量（轴角表示法），$\mathrm{Exp}(\cdot)$将其转换为旋转矩阵。
- 垂直偏移参数$p_z$：由于源动作数据本身可能存在漂浮或伪影，所以需要对运动整体进行垂直偏移修正。其中$z_{\mathrm{nom}}$是预先计算好的固定垂直偏移量，$p_z$则是可学习的额外垂直偏移。$\mathbf e_z$是世界坐标系的$z$轴单位向量。

可学习的参数化变换在名义变换的基础上，补偿了运动过程中出现的动态差异，例如动态平衡需求、接触噪声等。

## 4. 约束条件与损失函数

上层参数的可行域$\mathcal P$为优化器提供了约束条件

$$
\|\mathbf p^b_\mathrm{pos}\|_2\leq\delta_\mathrm{pos},\quad\|\mathbf p^b_\mathrm{ori}\|_2\leq\delta_\mathrm{ori},\quad|p_z|\leq\delta_z
$$

该约束条件避免了优化器对重定向参数的极端调整，确保重定向过程在物理和语义上保持相似性，不会产生荒谬的畸变。

损失函数对每个运动量分别计算再求和，其中位置、速度和角速度的损失项即为欧式距离的平方

$$
\ell^b_\mathbf x=\|\mathbf x^b_{\mathbf g_t}-\mathbf x^b_{\mathbf s_t}\|^2_2,\quad\ell^b_\mathbf v=\|\mathbf v^b_{\mathbf g_t}-\mathbf v^b_{\mathbf s_t}\|^2_2,\quad\ell^b_{\boldsymbol\omega}=\|\boldsymbol\omega^b_{\mathbf g_t}-\boldsymbol\omega^b_{\mathbf s_t}\|^2_2
$$

姿态损失项为

$$
\ell^b_\mathbf R=\|\mathrm{Log}((\mathbf R^b_{\mathbf s_t})^T\mathbf R^b_{\mathbf g_t})\|^2_2
$$

该式通过$(\mathbf R^b_{\mathbf s_t})^T\mathbf R^b_{\mathbf g_t}$计算参考姿态和实际姿态之间的相对旋转，再通过$\mathrm{Log}(\cdot)$将其转换为三维向量，并计算其长度的平方。

但是考虑到机器人与人类的关节自由度不匹配的情况，如人类的髋关节有三个自由度，但四足机器人的髋关节往往只有两个，限制所有方向上的旋转误差会让策略陷入困境。为此，ReActor提出可以将旋转误差分解为摆动（Swing）和扭转（Twist），其中扭转是绕用户指定轴的旋转，进而选择只惩罚摆动误差，忽略不可控的扭转误差。

最终总损失函数为

$$
\ell(\mathbf g_t-\mathbf s_t)=\sum_b(w_\mathbf x\ell^b_\mathbf x+w_\mathbf R\ell^b_\mathbf R+w_\mathbf v\ell^b_\mathbf v+w_{\boldsymbol\omega}\ell^b_{\boldsymbol\omega})
$$

# 四、强化学习

下层强化学习的目标是训练一个策略$\pi_\phi(\mathbf a_t|\mathbf o_t,\mathbf g_t)$来跟踪参考运动$\mathbf g_t$。其中$\mathbf a_t$为动作，$\mathbf o_t$为观测。

## 1. 动作空间

$$
\mathbf a_t:=(\mathbf a_t^\mathrm{pts},\mathbf w_t^\mathrm{rt})\quad\text{with}\quad\mathbf w^\mathrm{rt}_t:=(\mathbf f^\mathrm{rt}_t,\boldsymbol\tau^\mathrm{rf}_t)
$$

其中$\mathbf a_t^\mathrm{pts}$为关节目标位置，$\mathbf w_t^\mathrm{pt}$为根部辅助力/力矩，包括三个方向的力$\mathbf f^\mathrm{rt}_t$和三个方向的力矩$\boldsymbol\tau^\mathrm{rf}_t$。

根部辅助力/力矩通常作用于机器人的骨盆或躯干，相当于一个作弊器，帮助策略在训练初期完成高难度的动作，一方面让智能体能够进行充分的初始探索，另一方面产生有意义的轨迹数据为上层优化器提供反馈信号，避免训练崩溃。但辅助力/力矩只能用作推动训练的临时工具，需要引入惩罚让策略逐渐学会减少对其的依赖，惩罚的计算在奖励函数部分介绍。不过即便是在训练初期，我们也需要避免策略过度依赖辅助力/力矩

$$
\mathbf w^\mathrm{rt}_t:=\mathrm{sgn(\mathbf w^\mathrm{rt}_t)}\odot\max(0,\mathrm{abs}(\mathbf w^\mathrm{rt}_t)-d)
$$

该公式设置了一个阈值$d$，只有当输出的辅助力/力矩超过阈值时才会生效，这鼓励策略在非必要的情况下不使用辅助力/力矩。

## 2. 观测空间与初始化

$$
\mathbf o_t:=(h^\mathrm{rt}_t,\boldsymbol\theta^\mathrm{rt}_t,\mathbf v^\mathrm{rt}_t,\mathbf\omega^\mathrm{rt}_t,\mathbf q_t,\dot{\mathbf q}_t,\mathbf a_{t-1},\mathbf a_{t-2},\psi_t)
$$

- 根部高度$h^\mathrm{rt}_t$：感知是否离地。
- 重力投影$\boldsymbol\theta^\mathrm{rt}_t$：感知身体倾斜。
- 根部线速度$\mathbf v^\mathrm{rt}_t$和角速度$\mathbf\omega^\mathrm{rt}_t$：感知躯干状态。
- 关节位置$\mathbf q_t$和速度$\dot{\mathbf q}_t$：感知关节状态。
- 历史动作$\mathbf a_{t-1}$和$\mathbf a_{t-2}$：提供时序信息。
- 重定向阶段变量$\psi_t$：见后文。

重定向阶段变量$\psi_t$用于解决机器人的初始化问题。传统运动重定向方法使用参考状态初始化，直接将机器人的关节设置到参考运动的第一帧位置。但是在实际场景中，源角色的初始姿势与机器人的初始姿势可能完全不同，并且难以直接从参考运动中提取机器人的初始关节角度。

ReActor使用重定向阶段变量$\psi_t$实现了渐进式初始化。$\psi_t$将从$0$开始，并随时间线性增加到$1$。

- 当$\psi_t<1$时，参考运动暂停在第一帧，机器人的目标是从随机初始姿势逐渐靠近参考运动的第一帧，相当于热身阶段。
- 当$\psi_t=1$时，参考运动开始播放，真正进入重定向跟踪阶段。

跟踪奖励将乘以$\psi_t$进行缩放，避免在热身阶段给机器人不合理的惩罚。同时，只有$\psi_t=1$的数据才会被用于上层参数优化，避免初始化噪声干扰参数学习。

## 3. 自适应动作采样

由于运动数据集中不同片段的难度差异很大，如果均匀采样，策略会在容易的运动上花太多时间，而困难的运动又学不够。为此，ReActor为每个运动片段维护一个失败计数，根部位置误差或姿态误差超过一定阈值则触发回合失败。动作片段的采样概率正比于失败率。

## 4. 奖励函数

$$
r_t=r^\mathrm{tracking}_t+r^\mathrm{regularization}_t
$$

| 名称         | 奖励项                                                                                                               | 权重                   |
| ------------ | -------------------------------------------------------------------------------------------------------------------- | ---------------------- |
| **运动跟踪** | $r^\mathrm{tracking}_t$                                                                                              |                        |
| 根部水平位置 | $-\ell_{x,y}^{\text{rt}}$                                                                                            | $2.0$                  |
| 根部高度     | $-\ell_{z}^{\text{rt}}$                                                                                              | $10.0$                 |
| 根部姿态     | $-\ell_{\mathbf R}^{\text{rt}}$                                                                                      | $2.0$                  |
| 根部线速度   | $-\ell_{\mathbf v}^{\text{rt}}$                                                                                      | $0.5$                  |
| 根部角速度   | $-\ell_{\boldsymbol\omega}^{\text{rt}}$                                                                              | $0.5$                  |
| 刚体位置     | $-\ell_{\mathbf x}^{\text{b}}$                                                                                       | $2.0 · \psi_t$         |
| 刚体姿态     | $-\ell_{\mathbf R}^{\text{b}}$                                                                                       | $2.0 · \psi_t$         |
| 生存         | $1.0$                                                                                                                | $20$                   |
| **正则化**   | $r^\mathrm{regularization}_t$                                                                                        |                        |
| 关节力矩     | $-\|\boldsymbol\tau_t^{\text{jts}}\|_2^2$                                                                            | $1.0 \cdot 10^{-4}$    |
| 关节加速度   | $-\|\dot{\mathbf q}_t\|_2^2$                                                                                         | $1.0 \cdot 10^{-6}$    |
| 动作变化率   | $-\|\dot{\mathbf a}_t^{\text{jts}} - \dot{\mathbf a}_{t-1}^{\text{jts}}\|_2^2$                                       | $1.0 \cdot 10^{-2}$    |
| 动作变变化率 | $-\|\dot{\mathbf a}_t^{\text{jts}} - 2\dot{\mathbf a}_{t-1}^{\text{jts}} + \dot{\mathbf a}_{t-2}^{\text{jts}}\|_2^2$ | $1.0 \cdot 10^{-2}$    |
| 根部辅助力   | $-\|\mathbf{f}_t^{\text{rt}}\|_1$                                                                                    | $\psi_t \cdot 10^{-2}$ |
| 根部辅助力矩 | $-\|\boldsymbol\tau_t^{\text{rt}}\|_1$                                                                               | $\psi_t \cdot 10^{-2}$ |

# 五、总结

ReActor首次将双层优化引入运动重定向，实现了参数和策略的协同优化，并且允许稀疏的刚体语义对应，无需人为设计复杂参数，能够轻松支持形态差异巨大的机器人。未来研究方向包括：

- 时变参数化：允许重定向参数随时间变化，适应更复杂的运动。
- 自动化对应：减少人工输入，自动建立刚体语义对应。
- 复杂任务集成：将运动重定向与避障、操作、机器人自动设计等结合。
- 训练生成式模型：利用高质量重定向数据训练运动生成模型。

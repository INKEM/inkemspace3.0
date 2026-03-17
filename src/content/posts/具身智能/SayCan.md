---
title: 【SayCan】LLM+价值函数：以言为引，量力而行
date: 2025-10-19
summary: LLM分层决策的开山之作。
tags: [SayCan, LLM决策, LLM, 论文精读]
category: 具身智能
---

> 论文标题：Do As I Can, Not As I Say: Grounding Language in Robotic Affordances
>
> 论文发表时间：2022年8月

# 研究背景

大语言模型（Large Language Model，LLM）凭借其超大规模的参数量，从网上庞大的语料库中汲取了大量的知识，已经能够完成复杂的语言处理与生成任务。这样的能力对于机器人接收人类下达的自然语言作为指令完成任务至关重要，我们自然期望将LLM直接应用于机器人执行自然语言指令任务。

但是LLM仅仅基于上下文完成任务，无法直接获得来自物理世界的信息，包括机器人处于什么样的环境与状态，某个行动会对现实产生什么样的影响等等。LLM具有分析并响应自然语言指令的能力，即“Say”，但机器人能不能理解其响应，进而又能不能在现实世界中执行，即“Can”，LLM自身则无法保证。

_早期的LLM针对“饮料洒了”，GPT3回应“试试使用真空吸尘器”，但场景中可能没有真空吸尘器或机器人无法使用；LaMDA和FLAN的回复“要我找个真空吸尘器吗？”“对不起，我不是故意的”仅仅从对话意义上来看是合理的。_

![Pasted image 20251019195323](https://inkem-1306784622.cos.accelerate.myqcloud.com/blog/pic/Pasted%20image%2020251019195323.png)

为此，SayCan先给智能体训练一系列配备了语言描述的技能单元，规范LLM的“Say”使之输出机器人可以理解的低级技能序列，再学习与技能对应策略的价值函数作为衡量“Can”的指标。通过将LLM的“Say”和现实世界的“Can”结合起来作为机器人高层决策的依据，SayCan让机器人完成了简单的自然语言指令任务。

# SayCan：Do As I Can, Not As I Say

## 问题陈述

系统接收用户提供的自然语言指令$i$，描述机器人应该执行的任务，可以是长的、抽象的或模糊的。

智能体被赋予一个技能库$\Pi$，其中每个技能$\pi\in\Pi$执行一个简短的任务，并附带一个简短的语言描述$\mathscr l_\pi$和示能函数（affordance function）$p(c_\pi|s,\mathscr l_\pi)$，表示在状态$s$下语言描述为$\mathscr l_\pi$的技能可以被完成（complete，$c_\pi$）的概率。本质上示能函数是$p(c_\pi|s,\pi)$，使用$\mathscr l_\pi$主要是为了在形式上与LLM的输入输出保持统一。在RL中，示能函数可以是技能的价值函数，成功完成的奖励为1，否则为0。

_下图展示了不同状态下的示能空间$\{p(c_\pi|s,\mathscr l*\pi)\}*{\pi\in\Pi}$。当场景中出现苹果和红牛罐头时，拾取苹果和红牛罐头具有较高的价值，且距离近者价值更高。\_

![Pasted image 20251018224442](https://inkem-1306784622.cos.accelerate.myqcloud.com/blog/pic/Pasted%20image%2020251018224442.png)

LLM给出技能的语言描述$\mathscr l_\pi$是用户指令$i$的有效下一步的概率$p(\mathscr l_\pi|i)$，那么对于我们关注的某项技能可以被成功执行，并朝着完成指令的方向取得进展的概率$p(c_i|i,s,\mathscr l_\pi)$有

$$
p(c_i|i,s,\mathscr l_\pi)\propto p(c_\pi|s,\mathscr l_\pi)p(\mathscr l_\pi|i)
$$

其中$p(\mathscr l_\pi|i)$被称为任务基础（task-grounding），$p(c_\pi|s,\mathscr l_\pi)$被称为世界基础（world-grounding）。

## 连接LLM到机器人

一个原始的LLM能够合理地响应我们给出的指令，但它不知道如何以机器人可以理解的方式，即低级技能序列作为响应。一种方法是依赖提示词工程，但这仍然无法对其输出进行严格的约束。为此，SayCan没有让LLM直接生成技能文本，而是为给定的文本计算可能性。

首先，设计提示词，通过少样本学习来引导LLM，让LLM学会以特定的格式和结构来回答“How would you ...”，这是LLM后续给出合理概率打分的重要基础。

![Pasted image 20251019152111|500](https://inkem-1306784622.cos.accelerate.myqcloud.com/blog/pic/Pasted%20image%2020251019152111.png)

接下来，以用户指令$i$作为上文，给定一系列候选技能的语言描述$\mathscr l_\Pi$，LLM将计算每个候选技能描述$\mathscr l_\pi\in\mathscr l_\Pi$被预测为指令$i$的下文的概率$p(\mathscr l_\pi|i)$。如果该概率很高，则意味着这个技能与用户指令的关联性很强，因而可以自然而然地成为了衡量技能对完成指令的推动作用的指标。

遍历完第一步的候选动作后，LLM将选取概率最高的候选技能$\mathscr l_\pi=\mathrm{argmax}_{\mathscr l_\pi\in\mathscr l_\Pi}p(\mathscr l_\pi|i)$执行，并将其附加到指令$i$中用于对下一步候选动作进行打分，直至LLM认为指令已完成。

_直接使用概率进行计算会导致数值下溢，因此LLM实际上给出的是对数概率。_

![Pasted image 20251019153451](https://inkem-1306784622.cos.accelerate.myqcloud.com/blog/pic/Pasted%20image%2020251019153451.png)

## 策略训练

_本小节简要介绍策略训练的具体实现，非核心内容_

SayCan同时使用了基于BC-Z方法的行为克隆（Behavior Cloning，BC）和基于MT-Opt的RL来获取技能的策略。实验人员发现，BC策略在当前的数据集和实验条件下的成功率更高，但是BC策略在短期内的高成功率会使其训练得到的价值函数对失败边界的认识不足，相比之下RL策略训练出的价值函数更为可靠。所以最终，BC策略被选为执行器，而RL策略的价值函数则用于近似估计BC策略的示能函数。

奖励采用稀疏奖励函数，即技能执行成功为1，失败为0。关于成功的判定，实验人员采取了3人投票多数表决的方法，因为机器人判定难度太大。虽然人工方法效率较低，但是对于证明算法的可行性来说已经足够。

## SayCan

现在我们将LLM的Say与示能函数的Can结合。在每一步技能决策，LLM根据指令上文给出技能的任务基础$p(\mathscr l_\pi|i)$，示能函数根据机器人物理状态给出技能的世界基础$p(c_\pi|s,\mathscr l_\pi)$。最终二者相乘得到的概率（在工程实践中采取对数概率相加）是SayCan选择技能的依据

$$
\pi=\mathrm{argmax}_{\pi\in\Pi}p(c_\pi|s,\mathscr l_\pi)p(\mathscr l_\pi|i)
$$

_为了在合适的尺度下直观对比，论文图中LLM列出对数概率，示能函数列出原始概率_

![Pasted image 20251019160835|500](https://inkem-1306784622.cos.accelerate.myqcloud.com/blog/pic/Pasted%20image%2020251019160835.png)

技能选择完毕后，智能体执行对应的策略，LLM将对应的技能描述纳入指令，随后进行下一轮决策，直至SayCan选择了终止技能。整个SayCan算法的伪代码如下

![Pasted image 20251019180941](https://inkem-1306784622.cos.accelerate.myqcloud.com/blog/pic/Pasted%20image%2020251019180941.png)

# 成果与展望

论文针对实验采用了两个指标，均采用3人投票多数表决判定：

- 规划成功率：LLM针对自然语言指令选择的技能是否正确，无论其是否被成功执行；
- 执行成功率：整个SayCan系统是否真正成功地执行了所需指令。

最终，SayCan在模拟厨房环境下针对简单的自然语言指令实现了84%的规划成功率和74%的执行成功率（101次任务）。

_在寻找清理工具的指令下，SayCan对各个技能的打分_

![Pasted image 20251019210945](https://inkem-1306784622.cos.accelerate.myqcloud.com/blog/pic/Pasted%20image%2020251019210945.png)

实验的其他对比：

- 从实验室模拟厨房推广到其他真实厨房：规划成功率降低了3%，执行成功率降低了14%，问题主要在BC策略的迁移；
- 无示能函数：规划成功率降低了14%，毕竟物理环境和技能集合较为简单；
- 无LLM：仅在自然语言指令就是单个技能的情况有60%的执行成功率，其余情况完全不行。
- 不同LLM：更大的模型表现得更好，尤其是在更具挑战性的指令上体现的更明显。

SayCan的可行性得到了初步的证明，但也有很多可想而知的局限性：

- LLM本身的局限性与偏差将会继承到系统中；
- 系统的主要瓶颈在于单元技能的范围和能力；
- 系统不易对某一步偶然执行失败的情况作出反应。

论文指出下一步的工作在于：

- 将机器人在现实世界的经验用于改进LLM本身；
- 考虑结合价值函数以外的方式来计算示能函数；
- 研究自然语言是否适合作为机器人编程的范式；
- 开展更多将自然语言与机器人技术（规划、策略学习、人机交互等）的研究。

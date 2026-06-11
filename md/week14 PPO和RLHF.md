> edit time: 2026-06-08 00:22:44 PPO，RLHF的PPO，DPO
> edit time: 2026-06-08 12:20:33 IPO，KTO，GRPO，收敛性分析

本周一直在说的“目标”就是“损失函数”。

所有主流PO方法都是在优化KL正则化的目标，只是 $\text{Quality}$ 的定义不同：
$$
\max_{\pi} \mathbb{E}[\text{Quality}(\pi)] - \beta D_{KL}(\pi \| \pi_{\text{ref}})
$$


所有 KL 正则化目标的最优策略都有形式：
$$
\pi^*(y|x) \propto \pi_{\text{ref}}(y|x) \cdot f(y,x)
$$
即所有PO方法都是在对同一个先验 $\pi_{ref}$ 做指数倾斜 $\pi^* \propto \pi_{ref}e^{g/\beta}$ ，区别仅仅在于用什么能量/质量信号 $g(y)$ 。
## PPO-Penalty和PPO-Clip

**PPO-Penalty**目标
$$L^{KLPEN}(\theta)=\mathbb{E}\left[\frac{\pi_\theta(a|s)}{\pi_{\theta_{old}}(a|s)} A^{\pi_{\theta_{old}}} - \beta \cdot D_{KL}(\pi_{\theta_{old}}(\cdot | s) || \pi_\theta(\cdot | s)) \right]$$
~~KLPEN大概是KL Penalty的意思罢（无慈悲~~
设TRPO的KL约束为 $\delta$ ，则最优 $\beta^*$ 满足
$$\beta^*=\frac{1}{\alpha}=\sqrt{\frac{\nabla_\theta L^{CPI}(\theta^*)^T F^{-1} \nabla_\theta L^{CPI}(\theta^*)}{2\delta}}$$
其中 $F$ 是Fisher矩阵，$\alpha$ 是TRPO的最优步长。
****
**PPO-Clip**目标
$$L^{CLIP}(\theta)=\mathbb{E}[\min(r(\theta)A, \text{clip}(r(\theta),1-\epsilon,1+\epsilon)A)]$$
其中 $r(\theta)=\frac{\pi_\theta(a|s)}{\pi_{\theta_{old}}(a|s)}$ 是重要性比率，$\epsilon$ 通常取0.1或0.2。

clipping与KL约束的关系：若 $r(\theta) \in [1-\epsilon,1+\epsilon]$ 对所有 $(s,a)$ 成立，则
$$D_{KL}(\pi_{old}||\pi_\theta) \le \frac{\epsilon^2}{2(1-\epsilon)}$$
常用的 $\epsilon=0.2$ 对应KL上界约为 $0.025$ 。

PPO-Clip的梯度：
$$\nabla_\theta L^{CLIP}=\mathbb{E}[r(\theta) A \cdot \nabla_\theta \log \pi_\theta(a|s) \cdot \mathbf{1}[r \in \text{active region}]$$
在 $\theta=\theta_{old}$ 时退化为 $A \nabla_\theta \log \pi_\theta$ 。
其中，$\text{active region}$ 是clip不起作用的区域，即 $r(\theta) \in (-\infty,1-\epsilon) \cup (1+\epsilon, \infty)$ 。
直觉上，在clip范围之外的那些比例，对梯度影响为0。
## RLHF中的PPO

**RLHF目标**为
$$\max_\theta \mathbb{E}_{x \sim \mathcal{D},y \sim \pi_\theta(\cdot | x)}[R(x,y)] - \beta D_{KL}(\pi_\theta || \pi_{ref})$$
其中 $\pi_{ref}$ 是预训练模型。KL惩罚保持模型不偏离预训练太远，降低Reward Hacking问题。

KL正则化**RLHF目标的最优策略 $\pi^*$ 闭式解**为：
$$\pi^*(y|x)=\frac{1}{Z(x)} \pi_{ref}(y|x) \text{exp}(\frac{R(x,y)}{\beta})$$
即最优策略等于：参考模型乘以奖励的指数加权，再归一化。**“$\pi$乘$e$的$R$比$\beta$”**
模型不是一味追高奖励，而是"在参考模型附近"移动——奖励高的回答概率上升、低的下降，KL惩罚拦着它别偏离原模型太远（β越大越保守）。
碎碎念：
这个形式看起来像是将 $π_{ref}$ 对 $R$ 进行了一个Softmax加权，所以 $\beta$ 起到一个温度的作用。
- 温度趋于 $0$ 时，KL弱约束，$\pi^*$ 退化为只做reward最高的回答（贪婪策略）；
- 温度趋于 $\infty$ 时，KL强约束，$\pi^*$ 退化到先验预训练模型策略 $\pi_{ref}$ 。

其中分母这个归一化因子称为配分函数
$$Z(x)=\sum_y \pi_{ref}(y|x) \text{exp}(\frac{R(x,y)}{\beta})$$
它需要对所有的可能回答 $y$ 求和，但语言模型的回答空间过大、无法枚举。所以不考虑直接对闭式解的计算，而是以闭式解为指导，延伸出：
- 传统RLHF（先学奖励，再用PPO近似优化该目标）
- DPO/IPO/KTO/ORPO（利用闭式解把奖励消掉，直接从偏好数据训练策略）
这两大流派。

RLHF闭式解通过固定 $x$ ，优化RLHF目标（设 $\sum_y \pi(y)=1$）即Lagrangian得到：
$$\mathcal{L}=\sum_y \pi(y)R(y)-\beta \sum_y \pi(y) \log {\frac{\pi(y)}{\pi_{ref}(y)}} + \lambda(1-\sum_y \pi(y))$$
解法为对 $\pi(y)$ 求导。

当然也可以从 $\pi^*$ 反解出奖励：
$$R(x,y)=\beta \log \frac{\pi^*(y|x)}{\pi_{ref}(y|x)} + \beta \log Z(x)$$
这个式子说，拿到对齐后的 $\pi^*$ 就能读出奖励，$\log \frac{\pi^*}{\pi_{ref}}$ 正是对齐把回答 $y$ 相对预训练先验抬高/降低了多少，而不必单独训练Reward Model，对应DPO。
与此同时，$\beta \log Z(x)$ 只依赖 $x$ 而不依赖 $y$ ，这意味着奖励的绝对值没有意义，只有同一prompt内的奖励差有意义，这也对应着 $\log Z$ 在DPO中必然抵消。
****
eg1. 验证一下温度 $\beta$ 对RLHF最优解的奖励和先验模型的权衡，以及抬高/降低作用。
~~温度趋于0时，“温度太低，不鼓励创新和活泼”，模型趋于贪婪；
温度趋于无穷时，“温度太高，模型回归原始野性”（bushi。~~
假设prompt x = ”Explain quantum physics”，三个候选回答及其奖励为：

| 回答                                         | $R(x,y)$ | $\pi_{ref}(y\|x)$ |
| ------------------------------------------ | -------- | ----------------- |
| $y_1$: ”It’s complicated.”                 | 0.2      | 0.5               |
| $y_2$: ”Quantum physics studies...” (中等长度) | 0.8      | 0.3               |
| $y_3$: ”Quantum physics is...” (详细解释)      | 1.0      | 0.2               |

计算例如 $\beta=1$ 时，
$$\pi^*(y_1) \propto 0.5 \times e^{0.2/1} \approx 0.611$$
$$\pi^*(y_2) \propto 0.3 \times e^{0.8/1} \approx 0.668$$
$$\pi^*(y_3) \propto 0.2 \times e^{1.0/1} \approx 0.544$$
归一化因子 $Z=0.611+0.668+0.544=1.823$ ，之后归一化处理后得到
$$\pi^*(y_1)=0.335,\pi^*(y_2)=0.366,\pi^*(y_3)=0.298$$
对这个 $\beta=1$ 的例子，顺手从最优策略 $\pi^*$ 反推一下奖励 $R$ 可以得到:
$$R(y_1)=1 \times \log \frac{0.335}{0.5}+\log 1.823=0.200$$
相同流程有 $R(y_2) \approx 0.800, R(y_3) \approx 1.000$ 。
这也符合DPO能工作的原因：从 $\pi^*/\pi_{ref}$ 可以恢复奖励 $R$ 。

同理对于五组 $\beta$ （0, 0.5, 1.0, 2.0, $\infty$），可以解得以下的总结表：

| $\beta$            | $\pi^*(y_1)$ | $\pi^*(y_2)$ | $\pi^*(y_3)$ | $D_{KL}(\pi^*\|\pi_{\text{ref}})$ |
| ------------------ | ------------ | ------------ | ------------ | --------------------------------- |
| $\pi_{\text{ref}}$ | 0.500        | 0.300        | 0.200        | 0                                 |
| 2.0                | 0.415        | 0.337        | 0.248        | 0.016                             |
| 1.0                | 0.335        | 0.366        | 0.298        | 0.057                             |
| 0.5                | 0.201        | 0.401        | 0.398        | 0.201                             |
| $\to 0$            | 0            | 0            | 1            | $\infty$                          |

可以由总结表看到，
$\beta$ 足够大时，KL约束足够小，最优解分布完全与预训练分布 $\pi_{ref}$ 相同；
$\beta$ 较小时，最优解分布更倾向于刷高奖励，甚至于贪婪地只取最高奖励回答。
****
## DPO

**Bradley-Terry偏好模型** 给定两个回答 $y_w$ （winning）和 $y_l$ （losing），偏好概率为
$$P(y_w \succ y_l | x) = \sigma(R(x,y_w)-R(x,y_l))$$
其中 $\sigma(z)=\frac{1}{1+e^{-z}}$ 是sigmoid函数。
"偏好概率是奖励差取sigmoid"，这只是“奖励差越大，偏好越确定”的一个建模方式。

**DPO目标**的推导：
将RLHF最优策略的奖励表示 $R$ 代入Bradley-Terry模型，得到DPO目标。具体而言，
$$R(x,y)=\beta \log \frac{\pi^*(y|x)}{\pi_{ref}(y|x)} + \beta \log Z(x)$$
代入Bradley-Terry模型得
$$P(y_w \succ y_l) = \sigma(R(y_w)-R(y_l))$$
$$=\sigma \left(\beta \log \frac{\pi^*(y_w)}{\pi_{ref}(y_w)}-\beta \log \frac{\pi^*(y_l)}{\pi_{ref}(y_l)} \right)$$
“beta乘log pi比pi 的差值再取sigmoid“
关键观察：RLHF最优策略中仍然存在的 $\log Z$ 抵消了，也就是说这个概率只通过奖励的“相对值”而不是绝对值建模。谁奖励相对更高，相应地偏好概率也更高。这让DPO可以直接优化。
接下来我们用 $\pi_\theta$ 代替 $\pi^*$ 得到概率 $P_\theta(y_w \succ y_l)$ 。

当我们固定单个偏好对数据 $(x,y_w,y_l)$ 时，可以把概率式子看作关于 $\theta$ 的似然函数：
$$L(\theta)=P_\theta(y_w \succ y_l)$$
（设随机变量 $X$ 有概率密度函数 $f(x;\theta)$ ，
概率是 $P(X=x|\theta)=f(x;\theta)$ ，是关于 $x$ 的函数、$\theta$ 已知；
似然是 $L(\theta | x)=f(x;\theta)$ ，看作关于 $\theta$ 的函数、$x$ 已知。
似然：已知数据，反过来评价哪个参数更可能产生这个数据”）

从而我们想最大化所有偏好对的对数似然：
$$\max_\theta \sum_{(x,y_w,y_l)} \log P_\theta(y_w \succ y_l)$$
这也等价于最小化负对数似然。
当然，求和和期望是等价的（mini-batch近似期望），所以我们得到下述**DPO目标**
$$\mathcal{L}_{DPO}(\theta)=-\mathbb{E}_{(x,y_w,y_l) \sim \mathcal{D}} \left[ \log \sigma(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)})\right]$$
****
接下来推导一下单个偏好对的DPO梯度（整个数据集的DPO梯度是再取一个mini-batch）。
设
$$u=\beta \log \frac{\pi^*(y_w)}{\pi_{ref}(y_w)}-\beta \log \frac{\pi^*(y_l)}{\pi_{ref}(y_l)}$$
$$\mathcal{L}=- \log \sigma(u)$$
考虑到 $\sigma$ 的导数
$$\sigma'(u)=\sigma(u)(1-\sigma(u))$$
（顺便说一下，有恒等式：$\sigma(-u)=1-\sigma(u)$，即sigmoid函数关于 $(0,0.5)$ 中心对称）
则有 $-\log \sigma$ 的导数
$$\frac{d}{du}(-\log \sigma(u))=-\frac{\sigma'(u)}{\sigma(u)}=-(1-\sigma(u))$$
考虑到 $u$ 对 $\theta$ 的梯度
$$\nabla_\theta u=\beta \nabla_\theta \log \pi_\theta(y_w)-\beta \nabla_\theta \log \pi_\theta(y_l)$$
故有**单个偏好对的损失 $\mathcal{L}$ 对 $\theta$ 的梯度**
$$\nabla_\theta \mathcal{L}=-\beta(1-\sigma(u))(\nabla_\theta \log \pi_\theta(y_w)-\nabla_\theta \log \pi_\theta(y_l))$$
我们发现“梯度系数 $\beta(1-\sigma(u))$”有：
单个偏好对的损失取值 $1-\sigma(u)$ ，正是当前模型预测错误的概率。
预测越错（正确的减去错误的得到的 $u$ 越小），梯度越大。
梯度方向倾向于增加 $y_w$ 的概率、减小 $y_l$ 的概率。

可以证明，DPO的全局最优等于RLHF的最优：
若偏好数据由真实奖励 $R^*$ 和Bradley-Terry模型生成，且 $π_θ$ 有足够表达能力，则DPO的全局最优 $π^*_θ$ 等于RLHF 的最优策略。证明如下：
在期望意义下（对所有偏好对），最优条件是
$$\sigma(u_\theta)=P^*(y_w \succ y_l), 对所有 (y_w,y_l)$$
把Bradley-Terry模型：$P^*=\sigma(R^*(y_w)-R^*(y_l))$ 以及 $u_\theta$ 的定义代入上式，则有
$$\sigma \left( \beta \log \frac{\pi^*(y_w)}{\pi_{ref}(y_w)}-\beta \log \frac{\pi^*(y_l)}{\pi_{ref}(y_l)} \right)=\sigma(R^*(y_w)-R^*(y_l))$$
由于 $\sigma$ 严格单调，去掉 $\sigma$ 得
$$\beta \log \frac{\pi^*(y_w)}{\pi_{ref}(y_w)}-\beta \log \frac{\pi^*(y_l)}{\pi_{ref}(y_l)}=R^*(y_w)-R^*(y_l)$$
这是一个经典的可加性条件，它意味着存在某个常数 $c$ ，使得
$$\beta \log \frac{\pi_\theta(y)}{\pi_{ref}(y)}=R^*(y)+c, \forall y$$
这是因为：记 $f(y)=\beta \log \frac{\pi_\theta(y)}{\pi_{ref}(y)}$ ，则条件变形即 $f(y_w)-R^*(y_w)=f(y_l)-R^*(y_l)$ 。这个式子对任意的 $y_w$ 和 $y_l$ 都成立，所以这个差值是一个常数 $c$ （比如固定 $y_l$ 、则对任意 $y_w$ 都有左边等于右边固定的常数，所以左边应该是一个不依赖于 $y_w$ 的常数，与 $y_w$ 无关；同理有右边与 $y_l$ 无关，所以是常数 $c$），从而有 $f(y)=R^*(y)+c$ 即上式。
从该式容易推得
$$\pi_\theta(y) \propto \pi_{ref}(y)e^{R^*(y)/\beta}$$
这正是RLHF的最优策略形式。
****
顺便补充一个用于eg2数值计算分析的小的二级结论。对于这里的 $\nabla_\theta \log \pi_\theta(y_w)$ 和  $\nabla_\theta \log \pi_\theta(y_l)$ （像上一节的策略梯度），有简单计算方法：假设
$$\pi_\theta(y|x)=\text{softmax}(h_\theta(x))_y=\frac{\text{exp}(\phi_\theta(x,y))}{\sum_{y'}\text{exp}(\phi_\theta(x,y'))}$$
对于单个样本 $y$ ，固定 $x$ ，有
$$\log \pi_\theta(y)=\phi_\theta(y)-\log \sum_{y'} \text{exp}(\phi_\theta(y'))$$
对 $\theta$ 求梯度，有
$$\nabla_\theta \log \pi_\theta(y)=\nabla_\theta \phi_\theta(y)-\frac{\sum_{y'}  \text{exp}(\phi_\theta(y')) \nabla_\theta \phi_\theta(y')}{\sum_{y''} \text{exp}(\phi_\theta(y'')) }$$
注意到 $\pi_\theta(y')=\text{softmax}(h_\theta(y'))$ （分母求和用的是 $y''$ 没关系），则有梯度应为
$$\nabla_\theta \log \pi_\theta(y)=\nabla_\theta \phi_\theta(y)-\sum_{y'} \pi_\theta(y') \nabla_\theta \phi_\theta(y')$$
这正是期望形式
$$\nabla_\theta \log \pi_\theta(y)=\nabla_\theta \phi_\theta(y)-\mathbb{E}_{y' \sim \pi_\theta}[\nabla_\theta \phi_\theta(y')]$$
对于线性参数化的特例，有 $\phi_\theta(y)=\theta^T \psi(y)$ ，此时有 $\nabla_\theta \phi_\theta(y)=\psi(y)$ 。
假设 $\psi(y_w)$ 记作 $e_w$ （代表“赢家特征”），$\mathbb{E}_{y' \sim \pi_\theta}[\nabla_\theta \phi_\theta(y')]$ 记作 $\pi$ （代表“平均特征”），
则有
$$\nabla_\theta \log \pi_\theta(y_w)=e_w-\pi$$
同理有 $\nabla_\theta \log \pi_\theta(y_l)=e_l-\pi$ 。
这个小结论就像是在说，$\log \pi$ 的梯度值，是赢家那一项为1，减去平均策略。
****
eg2. 假设有Softmax策略，3个token：$y_w,y_l,y_{other}$ ，$\pi_\theta(y)=\frac{e^{\theta_y}}{\sum_z e^{\theta_z}}$ ，参数 $\theta=(\theta_w,\theta_l,\theta_o)$ ，初始 $\theta^{(0)}=(0,0,0)$ ，$\pi_\theta^{(0)}=(1/3,1/3,1/3)$ ，$\pi_{ref}=(1/3,1/3,1/3)$ ，$\beta=1$ ，学习率 $\alpha=0.5$ ，尝试对单个偏好对执行完整的DPO训练过程。
解：直觉上，$h_w$ 就是 $y_w$ 的 $\log \frac{\pi}{\pi}$ ，$u$ 是 $\beta(h_w-h_l)$ ，$\mathcal{L}$ 是 $-\log \sigma(u)$ 。
初始loss：
$$h_w=\log \frac{\pi_\theta(y_w)}{\pi_{ref}(y_w)}=\log \frac{1/3}{1/3}=0$$
同理有 $h_l=0$ 。则有 $u=\beta(h_w-h_l)=0,\mathcal{L}=-\log  \sigma(0)=0.693$ 。
梯度：（我们上述的二级小结论）
$$\nabla_\theta \log \pi_\theta(y_w)=e_w-\pi_\theta^{(0)}=(1,0,0)-(1/3,1/3,1/3)=(2/3,-1/3,-1/3)$$
$$\nabla_\theta \log \pi_\theta(y_l)=e_l-\pi_\theta^{(0)}=(0,1,0)-(1/3,1/3,1/3)=(-1/3,2/3,-1/3)$$
$$\nabla_\theta u=\beta(\nabla_\theta \log \pi_\theta(y_w)-\nabla_\theta \log \pi_\theta(y_l))=(1,-1,0)$$
$$\nabla_\theta \mathcal{L}=-(1-\sigma(0)) \cdot (1,-1,0)=(-0.5,0.5,0)$$
更新参数：
$$\theta^{(1)}=\theta^{(0)}-\alpha \nabla_\theta \mathcal{L}=(0,0,0)-0.5 \times (-0.5,0.5,0)=(0.25,-0.25,0)$$
新策略：
$$Z=e^{0.25}+e^{-0.25}+e^0=3.063$$
$$\pi^{(1)}=(1.284,0.779,1)/3.063=(0.419,0.254,0.327)$$
新loss：
$h_w=\log \frac{0.419}{1/3}=0.229$ ，$h_l=\log \frac{0.254}{1/3}=-0.272$ ，$u=1 \times (0.229 - (-0.272))=0.501$ ，
$$\mathcal{L}^{(1)}=-\log \sigma(0.501)=0.474$$
把上述数值整理为表格：

| 迭代  | $\pi(y_w)$ | $\pi(y_l)$ | $\pi(y_o)$ | $u$   | Loss  |
| --- | ---------- | ---------- | ---------- | ----- | ----- |
| 0   | 0.333      | 0.333      | 0.333      | 0     | 0.693 |
| 1   | 0.419      | 0.254      | 0.327      | 0.501 | 0.474 |

继续迭代到收敛，会发现 $\theta_w$ 慢慢增大、$\theta_l$ 慢慢减小，以相反方向增长，而与偏好无关的 $\theta_o$ 保持为0。极限情况下 $\pi(y_w) \rightarrow 1$ ，$\pi(y_l) \rightarrow 0$ ，但收敛变慢（梯度趋于0）。
## IPO和KTO

IPO只是一个DPO的补丁，效果也未必比DPO更好，简单介绍一下。
DPO 假设Bradley-Terry 模型完美成立。但实际中：
• 人类偏好有噪声
• 偏好可能不传递（$A \succ B$ 且 $B \succ C$ 但 $A \nsucc C$）
• 极端奖励差会导致数值问题
IPO 的想法：用更鲁棒的损失函数。
IPO的目标是：
$$
\mathcal{L}_{\text{IPO}}(\theta) = \mathbb{E} \left[ \left( \log \frac{\pi_\theta(y_w)}{\pi_{\text{ref}}(y_w)} - \log \frac{\pi_\theta(y_l)}{\pi_{\text{ref}}(y_l)} - \frac{1}{2\beta} \right)^2 \right]
$$
****
KTO将DPO/IPO的偏好“对”，放松到只需要单独标注的数据（pointwise）。从而可以只有点赞或点踩的数据。
KTO基于前景理论中的“损失厌恶”思想。
KTO的目标是：
$$
\mathcal{L}_{\text{KTO}}(\theta) = \mathbb{E}_{x,y} \left[ w(y) \cdot \left( 1 - v \left( \beta \log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)} - z_{\text{ref}} \right) \right) \right]
$$
## GRPO

![[Pasted image 20260608115135.png]]

这个要重点一些。
就像DPO一样，我们希望干掉critic模型 $V(s)$ ，这样大模型能用少得多的成本进行训练。于是，GRPO用组内相对奖励代替critic。
对每个prompt $x$ ，采样 $G$ 个回答 $\{y_1, \dots, y_G\}$ ，计算**标准化优势估计**：
$$\hat{A}_i=\frac{R(x,y_i)-\overline{R}}{\sigma_R}$$
其中奖励平均 $\overline{R}=\frac{1}{G}\sum_j R(x,y_j)$ ，标准差 $\sigma_R=\sqrt{\frac{1}{G} \sum_j(R(x,y_j)-\overline{R})^2}$ 。
则**GRPO的目标**为：
$$\mathcal{L}_{GRPO}(\theta)=-\mathbb{E}_{x,\{y_i\}}\left[ \frac{1}{G} \sum_{i=1}^G \min \left(r_i \hat{A}_i, \text{clip}(r_i,1-\epsilon,1+\epsilon)\hat{A}_i\right) \right]$$

为什么可以用这里的奖励平均 $\overline{R}$ ？
因为每个 $R_i$ 独立同分布从 $\pi(\cdot|x)$ 采样得到，是 $V(x)$ 的无偏估计，
所以它们的样本均值 $\overline{R}$ 同样也是 $V(x)$ 的蒙特卡洛无偏估计。

可以证明，**GRPO的优势估计是无偏的**：设 $R(x,y)$ 的期望为 $\mu$ ，可以证明 $\mathbb{E}[\hat{A}_i]=0$ ，即标准化后的相对优势期望为零，像真正的优势函数一样。证明如下：
$$\mathbb{E}[\hat{A}_i]=\mathbb{E} \left[ \frac{R_i-\overline{R}}{\sigma_R}\right]=\mathbb{E} \left[ \frac{R_i-\frac{1}{G} \sum_j R_j}{\sigma_R}\right]=\mathbb{E} \left[ \frac{R_i}{\sigma_R} \right]-\frac{1}{G} \sum_j \mathbb{E} \left[ \frac{R_j}{\sigma_R} \right]$$
由于 $R_i$ 和 $R_j$ 同分布，$\mathbb{E} \left[ \frac{R_i}{\sigma_R} \right]=\mathbb{E} \left[ \frac{R_j}{\sigma_R} \right]$ 。故有 $\mathbb{E}[\hat{A}_i]=0$ 得证。
****
eg3. 一个简单的数值例子
Prompt x: ”Write a poem about AI” ，采样 $G = 4$ 个回答，奖励：$R = [0.8, 0.6, 0.9, 0.5]$ ，尝试计算一下GRPO的优势估计。
$$\overline{R}= (0.8 + 0.6 + 0.9 + 0.5)/4 = 0.7$$
$$σ_R =\sqrt{\frac{1}{4}((0.8 − 0.7)^2 + (0.6 − 0.7)^2 + (0.9 − 0.7)^2 + (0.5 − 0.7)^2)} =\sqrt{0.025} = 0.158$$ 标准化优势应为：
$$\hat{A}=(0.8-0.7,0.6-0.7,0.9-0.7,0.5-0.7)/0.158=(0.63,-0.63,1.26,-1.26)$$
优势估计无偏可以通过把各项加到一起验证： $\sum_i \hat{A}_i=0$ 。
对于上述的标准化优势，我们可以看到 $y_3$ 最好、$y_4$ 最差，这和原始奖励的趋势是一致的。GRPO会增加 $y_3$ 的概率，减小 $y_4$ 的概率。
所以GRPO压根就是直接使用了原始奖励 $R$ ，只是标准化了一下。
****
对GRPO的局限性的讨论和改进尝试可以看DAPO和GSPO。

顺便补充一些Deepseek V4论文中的讨论。
根据论文第5.1节说明：
"Following pre-training, we conducted a post-training phase to yield the final models of DeepSeek-V4 series. Although the training pipeline largely mirrored that of DeepSeek-V3.2, **a critical methodological substitution was made: the mixed Reinforcement Learning (RL) stage was entirely replaced by On-Policy Distillation** (OPD; Gu et al., 2024; Lu and Lab, 2025)."

既然纯粹的RL pipeline被替换成了对多个领域首先各自GRPO训练专家模型、然后再把它们OPD蒸馏到一起，看来是后者能够有更高的提升了。
1. 单专家训练用GRPO（说明它够用）
2. 多专家融合改用OPD（说明GRPO不适合这个场景）
3. 难验证任务用GRM（说明GRPO依赖的组内比较在难验证任务上可能不够）
****
## 收敛性分析

**定理 13.1 (PPO 的近似单调改进)**. 设 $\epsilon$ 是 clip 参数, $\delta = \frac{\epsilon}{1-\epsilon}$. 若每次更新满足 $D_{KL}(\pi_{\text{old}} \| \pi_{\text{new}}) \leq c\delta^2$ (某常数 $c$), 则:
$$
J(\pi_{\text{new}}) \geq J(\pi_{\text{old}}) - O(\delta^2)
$$
即策略改进是近似单调的。
**证明思路.** Step 1: 回顾 Kakade-Langford 界
$$
J(\pi') \geq L_{\pi}(\pi') - C \cdot D_{\text{KL}}^{\text{max}}(\pi \| \pi')
$$
Step 2: Clip 限制了 KL
由 Clip 机制, $r = \pi' / \pi \in [1 - \epsilon, 1 + \epsilon]$.
单样本 KL: $KL(\pi \| \pi') = -\log r \leq \epsilon$ (当 $r$ 接近边界时)
Step 3: 代入得到界
$$
J(\pi') \geq L_{\pi}(\pi') - O(\epsilon)
$$
由于 $L_{\pi}(\pi') \geq L_{\pi}(\pi) = J(\pi)$ (clip 后的代理目标仍然非负改进), 我们有近似单调性。 $\square$
****
**定理 13.2 (DPO 的收敛性——凸优化视角)**. DPO 损失函数是 $\theta$ 的非凸函数, 但在 log-linear 策略参数化下, 有局部收敛保证。
**证明思路.** Step 1: DPO 损失的 Hessian
$$
\mathcal{L} = -\log \sigma(u), \quad \text{其中 } u = \beta(h_w - h_t), \quad h = \log \frac{\pi_{\theta}}{\pi_{\text{ref}}}
$$
$$
\nabla_{\theta}^2 \mathcal{L} = \beta^2 \sigma(u)(1 - \sigma(u))(\nabla_{\theta} h_w - \nabla_{\theta} h_t)(\nabla_{\theta} h_w - \nabla_{\theta} h_t)^T + \ldots
$$
Step 2: 局部凸性条件
当 $|u|$ 不太大时 $(\sigma(u)(1 - \sigma(u))$ 远离 0), Hessian 的主项是半正定的。
Step 3: 收敛速率
在局部凸区域, 梯度下降以 $O(1/t)$ 速率收敛。
在非凸区域, 可能收敛到局部最优或鞍点。 $\square$

## 10 RLHF的最优策略闭式解

从优化的视角推导一下KL正则化RL的闭式解，即RLHF最优策略。
考虑以下目标：
$$\max_{\pi(\cdot | x)} \mathbb{E}_{y \sim \pi(\cdot | x)} [R(y|x)] - \beta D_{KL}(\pi(\cdot|x) || \pi_{ref}(\cdot|x)), \ s.t. \ \sum_{y} \pi(y|x)=1$$
解题备注：
- 目标里面每一处参数都强调了“给定 $x$”，这是有**依赖关系**的。
- 另外 $\pi(\cdot|x)$ 这个写法是强调“**概率分布**”这个整体，而 $R(y|x)$ 则强调输出恰好为**特定值** $y$。
下述用Lagrange乘子法求闭式解。

(a) 我们前述目标定义中说明了每一处参数都需要强调“给定$x$”。但在下述推导中，为了书写方便起见，我们在中间过程中暂时省略 “$|x$“，而只写“输出为特定值 $y$”。则
$$D_{KL}=\sum_y \pi(y)\log \frac{\pi(y)}{\pi_{ref}(y)}$$
则目标应为
$$\max_{\pi} \sum_y \pi(y)R(y)-\beta \sum_y \pi(y)\log \frac{\pi(y)}{\pi_{ref}(y)}$$
(b) 对于上述优化目标和约束，对应的Lagrangian应为
$$\mathcal{L}(\pi,\lambda)=\sum_y \pi(y)R(y)-\beta \sum_y \pi(y) \log \frac{\pi(y)}{\pi_{ref}(y)}+\lambda(1-\sum_y \pi(y))$$
(c) $\mathcal{L}$ 对某个特定的 $y$ 求偏导（顺便注意 $\frac{\partial (\pi(y)\log \pi(y))}{\partial \pi(y)}=\log \pi(y)+1$），则求和式 $\sum_y$ 中其余项求导都为0，只有包含这个特定的 $y$ 的项对求导有影响。令求偏导的结果等于0则有
$$R(y)-\beta \log \pi(y)-\beta +\beta \log \pi_{ref}(y)-\lambda=0$$
(d) 由上式解出 $\pi(y)$ 得
$$\pi(y)=\pi_{ref}(y)\text{exp}\left(\frac{R(y)-\beta-\lambda}{\beta}\right)=\pi_{ref}(y) \cdot e^{-1-\lambda/\beta} \cdot e^{R(y)/\beta}$$
记与 $y$ 无关的部分 $e^{-1-\lambda/\beta}=1/Z(x)$ ，则推得**闭式解**
$$\pi^*(y|x)=\frac{1}{Z(x)}\pi_{ref}(y|x) \text{exp}(R(y|x)/\beta)$$
“$\pi$乘$e$的$R$比$\beta$”
(e) 由归一化条件 $\sum_y\pi^*(y|x)=1$ ，闭式解两侧对 $y$ 求和解得
$$Z(x)=\sum_y \pi_{ref}(y|x) \text{exp}(R(y|x)/\beta)$$
(f) 反解Reward。对(d)求得的闭式解两边取 $\log$ ，整理得到
$$R(y|x)=\beta \log \frac{\pi^*(y|x)}{\pi_{ref}(y|x)}+\beta \log Z(x)$$
该式子用于DPO，代入BT模型的奖励差以消去 $Z(x)$ 。
(g) 手算 $\pi^*$ 的物理意义。
考虑到 $\pi^* \propto \pi_{ref}(y)e^{R(y)/\beta}$ ，正比系数为 $1/Z(x)$ ，则计算 $\pi^*$ 时应该首先求得两个未归一化的 $\pi_{ref}(y_i)e^{R(y_i)/\beta}$ ，然后再归一化得到最优策略。
当 $\beta=0.1$ 时，派乘e的R比$\beta$值为 $(0.5 \cdot e^{1/0.1},0.5 \cdot e^{0/0.1})=(0.5e^{10},0.5)$ ，归一化得到 $(1,0)$ 。
当 $\beta=1$ 时，值为 $(0.5e^1,0.5e^0)=(1.359,0.5)$ ，归一化得到 $(0.731,0.269)$ 。
当 $\beta=10$ 时，值为 $(0.5e^{0.1},0.5e^0)$ ，归一化得到 $(0.525,0.475)$ 。
(h) $\beta$ 小的时候结果偏向奖励更高的，$\beta$ 大的时候结果与参考策略更相近。$\beta$ 控制奖励与KL散度约束（分布变化）的权衡。
## 9 GRPO的相对优势与零均值性

(a) $G=4,R=0.8,0.6,0.9,0.5$ ，则
$$\overline{R}=0.7$$
$$\sigma_R=\sqrt{0.025} \approx 0.158$$
$$\hat{A}_i=(R_i-\overline{R})/\sigma_R$$
则有
$$\hat{A}=(0.632,-0.632,1.265,-1.265)$$
(b) 从数值上有 $\sum_i \hat{A_i}=0$ 。而证明中有
$$\sum_i (R_i - \overline{R})=\sum_i R_i - G \overline{R}=G \overline{R}-G\overline{R}=0$$
(c) 因为 $R_i$ 是从 $y \sim \pi(\cdot|x)$ 分布独立采样得到的奖励，且
$$\mathbb{E}[\overline{R}]=\frac{1}{G} \sum_{i=1}^G \mathbb{E}[R_i]=V(x)$$
故 $\overline{R}$ 为 $V(x)=\mathbb{E}_{y \sim \pi}[R(x,y)]$ 的无偏Monte Carlo估计。 
(d)

| 维度       | PPO（用 critic Vϕ(s)）                        | GRPO（用群体均值 Rˉ）                         |
| -------- | ------------------------------------------ | -------------------------------------- |
| **训练成本** | 需额外训练一个 critic 网络，增加计算和内存                  | 无需 critic，节省训练开销                       |
| **样本效率** | 单 prompt 采 1 个 y 即可（借助 critic 估计 baseline） | 每个 prompt 需采 G≥2 个 y 才能获得有意义的 baseline |
| **适用任务** | 适合小模型、单次采样快，critic 容易训练                    | 适合 LLM 生成（一次生成 G 个回答并行），避免 critic 偏差   |
## 8 PPO-Clip

设 $\epsilon=0.2$ ，则clip区间为 $[0.8,1.2]$ ，设 $L^{CLIP}=\min(rA,\text{clip}(r,0.8,1.2) \cdot A)$ 。
(a)(b) 填表。

| 样本  | $r$ | $A$ | $rA$ | $\text{clip}(r)⋅A$ | $L^{CLIP}$ | 激活/未激活 |
| --- | --- | --- | ---- | ------------------ | ---------- | ------ |
| 1   | 1.5 | +2  | 3    | 2.4                | 2.4        | 激活     |
| 2   | 0.6 | +1  | 0.6  | 0.8                | 0.6        | 未激活    |
| 3   | 1.5 | -1  | -1.5 | -1.2               | -1.5       | 未激活    |
| 4   | 0.5 | -2  | -1   | -1.6               | -1.6       | 激活     |
| 5   | 1.0 | +1  | 1    | 1                  | 1          | 未激活    |

(c)
当 $A>0$ 时，$r > 1+\epsilon=1.2$ 时clip才激活。如样本1激活，而样本2和5都未激活。
当 $A < 0$ 时，$r < 1-\epsilon=0.8$ 时clip才激活。如样本4激活，而样本3未激活。
(d)
目标 $L^{CLIP}$ 不发生变化。
在 $r>1+\epsilon$ 区域，梯度为0。

## 7 GAE的几何级数化简

回忆TD误差形式为 $\delta_t=r_t+\gamma V(s_{t+1})-V(s_t)$ ，把若干个 $\delta$ 累加，就得到 $n$ 步优势估计 $\hat{A}_t^{(n)}=\sum_{l=0}^{n-1} \gamma^l \delta_{t+l}$ 。GAE把所有 $n$ 步估计做加权平均，权重按 $\lambda$ 几何衰减。（$1-\lambda$ 是归一化常数）：
$$\hat{A}_t^{GAE(\lambda)}=(1-\lambda)\sum_{n=1}^{\infty}\lambda^{n-1} \hat{A}_t^{(n)}$$
求证上述GAE的定义式等价于
$$\hat{A}_t^{GAE(\lambda)}=\sum_{l=0}^{\infty} (\gamma\lambda)^l \delta_{t+l}$$
(a) 首先把 $n$ 步优势估计的定义式代入GAE的定义式，有
$$\hat{A}_t^{GAE(\lambda)}=(1-\lambda)\sum_{n=1}^{\infty}\lambda^{n-1} \sum_{l=0}^{n-1} \gamma^l \delta_{t+l}$$
(b) $\delta_{t+l}$ 在双重求和条件 $1 \le n < +\infty,0 \le l \le n-1$ 中包含以下的项：
t
t t+1
t t+1 **t+2**
...
t t+1 **t+2** ... t+n-1
...
从上往下 $n$ 的值递增，从左往右 $l$ 的值递增。
对于每个固定的 $l$ ，$\delta_{t+l}$ 出现在 $n-1 \ge l$ 即 $n \ge l+1$ 的项中。例如上述 $l=2$ 这一列，$\delta_{t+2}$ 出现在 $n \ge 3$ 这些行中（除了前两行之外的行）。
故交换后的双重求和条件为 $0 \le l < +\infty, l+1 \ge n < +\infty$ 。交换求和顺序得
$$\hat{A}_t^{GAE(\lambda)}=(1-\lambda)\sum_{l=0}^{\infty}\gamma^l\delta_{t+l} \sum_{n=l+1}^{\infty}\lambda^{n-1}$$
(c) 计算内层几何级数。
设 $m=n-1$ ，则有
$$\sum_{m=l}^{\infty} \lambda^m=\lim_{x  \rightarrow \infty} \frac{\lambda^l(1-\lambda^x)}{1-\lambda}=\frac{\lambda^l}{1-\lambda}(0 < \lambda < 1)$$
(d) 接下来把内层几何级数代入，消去 $1-\lambda$ ，即得到所求证的
$$\hat{A}_t^{GAE(\lambda)}=\sum_{l=0}^{\infty} (\gamma\lambda)^l \delta_{t+l}$$
(e) 极限检查。把 $\lambda=0$ 和 $\lambda=1$ 分别代入简洁形式，发现：
$\lambda=0$ 时，$\hat{A}=\delta_t$ 。（因为这一项的 $l=0$ ， $\lambda$ 没啥用了）
这正对应于单步TD，低方差高偏差。
$\lambda=1$ 时，$\hat{A}=\sum_{l=0}^{\infty} \gamma^l \delta_{t+l}$ ，可以证明 $\hat{A}=G_t-V(s_t)$ 。
这正对应于MC偏差，高方差无偏差。
下证 $\hat{A}=G_t-V(s_t)$ 。首先代入 $\delta_{t+l}$ 的定义，有
$$\hat{A}_t = \sum_{l=0}^\infty \gamma^l(r_{t+l}+\gamma V(s_{t+l+1})-V(s_{t+l}))$$
$$\hat{A}_t=\sum_{l=0}^\infty \gamma^lr_{t+l}+\sum_{l=0}^\infty \gamma^{l+1}V(s_{t+l+1})-\sum_{l=0}^\infty \gamma^l V(s_{t+l})$$
设 $k=l+1$ 代入第二个和式，并将第三个和式拆成第一项和剩余和式。
$$\hat{A}_t=\sum_{l=0}^\infty \gamma^lr_{t+l}+\sum_{k=1}^\infty \gamma^kV(s_{t+k})-V(s_t)-\sum_{k=1}^\infty \gamma^kV(s_{t+k})$$
$$\hat{A}_t=\sum_{l=0}^\infty \gamma^lr_{t+l}-V(s_t)$$
故 $\hat{A}_t=G_t-V(s_t)$ 得证。
## 6 策略梯度定理的log-trick

参数化策略 $\pi_\theta(a|s)$ ，目标是最大化期望回报：
$$J(\theta)=\mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^\infty \gamma^tr_t\right]=\mathbb{E}_{s_0}[V^{\pi_\theta}(s_0)]$$
策略梯度定理写作
$$\nabla_\theta J(\theta)=\mathbb{E}_{s \sim d^\pi,a \sim \pi_\theta}[\nabla_\theta \log \pi_\theta(a|s) \cdot Q^\pi(s,a)]$$
其中 $\nabla_\theta \log \pi$ 是log-trick的产物。
(a) 设 $f(\theta)>0$ 是可微函数。证明 $\nabla_\theta \log f(\theta)=\nabla_\theta f(\theta) / f(\theta)$ 。
设 $u=f(\theta)$ ，则 $\nabla_\theta \log f(\theta)=\frac{d \log u}{du} \nabla_\theta=\nabla_\theta f(\theta)/f(\theta)$ 。
(b) 代入 $f=\pi_\theta(a|s)$ ，则有
$$\nabla_\theta \pi_\theta(a|s)=\pi_\theta(a|s) \cdot \nabla_\theta \log \pi_\theta(a|s)$$
(c) 考虑期望
$$\mathbb{E}_{a \sim \pi_\theta}[h(a)]=\sum_a \pi_\theta(a|s)h(a)$$
对 $\theta$ 求导，则应该等于对每一项的求导再加起来，有
$$\nabla_\theta \mathbb{E}[h(a)]=\sum_a [\nabla_\theta \pi_\theta(a|s)]h(a)$$
(d) 把(b)代入(c)，即利用 $\nabla \log \pi$ 替换 $\nabla \pi/\pi$ ，这就是log-trick。得到
$$\nabla_\theta \mathbb{E}_{a
 \sim \pi_\theta}[h(a)]=\mathbb{E}_{a \sim \pi_\theta}[\nabla_\theta \log \pi_\theta(a|s) \cdot h(a)]$$
观察这个式子，我们发现左边是一个抽样分布的期望求梯度（并不能采样估计），右边是一团东西的期望（可采样估计）。我们把不能采样估计的形式转化成了可采样估计的形式。
(e) 为什么不直接用 $\nabla \pi/\pi$ 而要用 $\nabla \log\pi$ ？
因为当 $\pi(a|s)\rightarrow 0$ 时，左边的 $\nabla \pi / \pi$ 的分母过小，必须除以这个小分母、容易溢出。而右边的 $\nabla \log \pi$ 则是有限值，可以通过先算 $\log \pi$ （有限、幅度适中的数）然后再自动微分得到。
(f) 考虑softmax策略 $\pi_\theta(a)=\frac{e^{\theta_a}}{\sum_b \theta_b}$ ，假设有3个动作。证明
$$\frac{\partial \log \pi_\theta(a)}{\partial \theta_c}=\mathbf{1}[a=c]-\pi_\theta(c)$$
即
$$\nabla_\theta \log \pi_\theta(a)=e_a-\pi$$
其中 $e_a$ 是第 $a$ 个标准基向量（即第 $a$ 个元素为1、其余元素为0），$\pi=(\pi_1,\pi_2,\pi_3)^T$ 。
下证。
首先对softmax取 $\log$ ，得到
$$\log \pi_\theta(a)=\theta_a - \log \sum_b e^{\theta_b}$$
接下来关于 $\theta_c$ 求偏导，得到
$$\frac{\partial \log \pi_\theta(a)}{\partial \theta_c}=\mathbf{1}[a=c]-\frac{e^{\theta_c}}{\sum_b e^{\theta_b}}=\mathbf{1}[a=c]-\pi_\theta(c)$$
整理为梯度即为
$$\nabla_\theta \log \pi_\theta(a)=e_a-\pi$$
这个式子是softmax取log梯度所一定满足的性质，很常用。
(g) 假设 $\pi=(0.5,0.3,0.2)$ ，则这三个动作各自的score向量应该为：
$$e_1-\pi=(1,0,0)-(0.5,0.3,0.2)=(0.5,-0.3,-0.2)$$
$$e_2-\pi=(0,1,0)-(0.5,0.3,0.2)=(-0.5,0.7,-0.2)$$
$$e_3-\pi=(0,0,1)-(0.5,0.3,0.2)=(-0.5,-0.3,0.8)$$
验证
$$\mathbb{E}_{a \sim \pi}[\nabla \log \pi]=\sum_a \pi_a(e_a-\pi)$$
$$=0.5 \times (0.5,-0.3,-0.2)+0.3 \times (-0.5,0.7,-0.2)+0.2 \times (-0.5,-0.3,0.8)=(0,0,0)$$
得证。
## 5 Q-learning和SARSA

(a) SARSA：
$$Q(s_t,a_t) \leftarrow Q(s_t,a_t)+\alpha[r_t + \gamma Q(s_{t+1},a_{t+1})-Q(s_t,a_t)]$$
初始时 $Q_{2\times2}$ 为全0。
考虑更新后的 $Q(s_1,a_1)$：
$$Q(s_1,a_1) \leftarrow 0+0.5 \times [1+0.9 \times 1 - 0]=0.95$$
注意它用到的是 $Q(s_t,a_t)$ 和 $Q(s_{t+1},a_{t+1})$ 两个表项。
(b) Q-learning：
$$Q(s_t,a_t) \leftarrow Q(s_t,a_t)+\alpha[r_t + \gamma \max_{a'}Q(s_{t+1},a') -Q(s_t,a_t)]$$
为了计算 $\max$ 需要比较 $Q(s_{2},a_{1})$ 和 $Q(s_2,a_2)$ 的大小，发现前者更大。故
$$Q(s_1,a_1) \leftarrow 0 + 0.5 \times [1+0.9 \times 2 - 0]=1.4$$
(c) 相差 $1.4-0.95=0.45$ 。发现Q-learning给出更大的更新，因为它用的是 $Q$ 值更大的表项。
(d) 
SARSA是on-policy的，使用的是实际采样得到的下一个动作 $a_{t+1}$ ，**评估的是行为策略**本身。
Q-learning是off-policy的，因为用了 $\max$ ，所以**评估的是贪婪策略**，而不是行为策略。
还可以看到，Q-learning的 $\max$ 会带来系统性过估计，总是高估 $Q$ 。
## 4 baseline降低方差

考虑单状态，标量参数。
设 $\text{score } \psi = \nabla_\theta \log \pi_\theta(a|s)$ ，
回报为 $G$ ，回报的期望和方差分别为 $\mu$ 和 $\sigma^2$ ，
策略梯度估计 $\hat{g}=\psi \cdot G$ ，加baseline后 $\hat{g}_b=\psi \cdot (G-b)$ ，
假设score $\psi$ 与回报 $G$ 独立，score有 $\mathbb{E}[\psi]=0$ ，$\mathbb{E}[\psi^2]=I$ 。
(a) 证明添加baseline有期望不变性。
$$\mathbb{E}[\hat{g_b}]=\mathbb{E}[\hat{g}]-\mathbb{E}[\psi \cdot b]=\mathbb{E}[\hat{g}]-b \mathbb{E}[\psi]=\mathbb{E}[\hat{g}]$$
(b) 推导无baseline的方差形式。
$$\text{Var}[\hat{g}]=\mathbb{E}[\psi^2G^2]-(\mathbb{E}[\psi G])^2$$
考虑到 $\psi$ 和 $G$ 独立，也有 $\psi^2$ 与 $G^2$ 独立。则有
$$\text{Var}[\hat{g}]=\mathbb{E}[\psi^2]\mathbb{E}[G^2]-(\mathbb{E}[\psi]\mathbb{E}[G])^2$$
其中 $\mathbb{E}[\psi]=0$ ，$\mathbb{E}[\psi^2]=I$ ，$\mathbb{E}[G^2]=\text{Var}[G^2]+(\mathbb{E}[G])^2=\sigma^2+\mu^2$ 。则有
$$\text{Var}[\hat{g}]=I(\sigma^2+\mu^2)$$
(c) 推导有baseline的方差形式。
加baseline后，$G-b$ 的期望为 $\mu-b$ ，方差仍为 $\sigma^2$ 。
则有
$$\text{Var}[\hat{g}_b]=I(\sigma^2+(\mu-b)^2)$$
(d) 把(c)看作关于 $b$ 的二次函数，使得方差最小的 $b^*$ 显然应有 $b^*=\mu$ 。
根据状态值函数的定义（期望累积折扣回报），$V=\mathbb{E}[G_t]=\mu$ 。故有 $b^*=\mu=V$ 。
此时的最小方差应为 $\text{Var}[\hat{g_{b^*}}]=I\sigma^2$ 。
(e) 我们发现从(b)到(c)，期望有了二次程度的下降。
假设 $\mu=100,\sigma=10,I=1$ ，则加baseline前
$$\text{Var}[\hat{g}]=1 \times (10^2+100^2)=10100$$
加baseline后，有 $b^*=\mu=100$ ，
$$\text{Var}[\hat{g}_b]=1 \times (100+(10^2-100)^2)=100$$
方差降低了 $10100/100=101$ 倍。
Critic 之所以学状态值函数V ，就是因为V 是最优baseline。
用A(s, a) = Q(s, a) − V (s) 替换Q，相当于把梯度信号零均值化——只保留” 这个动作比平均好多少”，丢掉与动作无关的公共项V 。(e) 中” 方差降101倍” 就是这个零均值化的威力。
为什么baseline 只能依赖状态s、不能依赖动作a？因为否则(a) 中” 期望不变” 会被破坏。
## 3 Monte Carlo评估

Monte Carlo是用累积折扣回报 $G_t$ 来估计期望累积折扣回报（即状态-值函数） $V(s)$ 。
一条轨迹为：（$\gamma=0.5$）
$$s_1 \xrightarrow{r_1=1} s_2 \xrightarrow{r_2=2} s_3 \xrightarrow{r_3=0} s_1 \xrightarrow{r_4=3} 终止.$$
其中状态 $s_1$ 被访问了两次，分别在 $t=1$ 和 $t=4$ 。
(a) 根据累积折扣回报return的定义，计算
$$G_4=3$$
$$G_3=0+0.5 \times 3=1.5$$
$$G_2=2+0.5 \times 1.5=2.75$$
$$G_1=1+0.5 \times 2.75=2.375$$
(b) Every-visit MC：把每次访问 $s$ 的 $G_t$ 都拿来平均。

| 状态    | 访问发生在   | $V(s)$ 的every-visit估计 |
| ----- | ------- | --------------------- |
| $s_1$ | $t=1,4$ | $(2.375+3)/2=2.6875$  |
| $s_2$ | $t=2$   | $2.75$                |
| $s_3$ | $t=3$   | $1.5$                 |

(c) First-visit MC：每条轨迹中只用首次访问的 $G_t$ 来估计值函数 $V(s)$ 。

| 状态    | 访问发生在   | $V(s)$ 的first-visit估计 |
| ----- | ------- | --------------------- |
| $s_1$ | $t=1,4$ | $2.375$               |
| $s_2$ | $t=2$   | $2.75$                |
| $s_3$ | $t=3$   | $1.5$                 |

其中状态 $s_1$ 的估计与Every-visit不同。因为该状态被访问多次，**在单条轨迹内**，First-visit只使用第一次访问到的 $G_t$ ，而Every-visit使用每次访问到的 $G_t$ 的平均。
（注意，不同轨迹应该用增量平均公式更新）

(d) 假设第二条轨迹有 $s_1$ 的return $G^{(2)}=5$ 。则有
$$V_{new}(s_1)=V_{old}(s_1)+\frac{1}{n}(G^{(2)}-V_{old}(s_1))$$
$$=2.375+\frac{1}{2} \times (5-2.375)=3.6875$$
(e) 对比MC和TD。MC必须等到整个episode结束才能更新（因为整个episode结束才能知道所有的 $G_t$ ），而 TD(0) 每步就能更新。
例如机器人在无边界的二维连续平面上执行任务，永不终止，这种情况MC不能用、而TD可以用。
## 2 值迭代

考虑2状态MDP $\{A,B\}$ ，$\gamma=0.5$ ，单动作（即策略评估，策略固定，不需要考虑策略的改进。如果每个状态只有一个动作，那么策略自然固定。）
- 状态A：奖励2，确定转到B。
- 状态B：奖励0，确定转到A。
Bellman更新即：
$$V_{k+1}(A)=2+0.5 V_{k}(B)$$
$$V_{k+1}(B)=0+0.5V_k(A)$$
(a) 取不动点，有联立方程。
$$V(A)=2+0.5V(B),V(B)=0.5V(A)$$
解得
$$V^*(A)=\frac{8}{3},V^*(B)=\frac{4}{3}$$
(b) 从 $V_0=(0,0)$ 开始，逐步代入Bellman更新公式，有

| $k$ | $V_k(A)$ | $V_k(B)$ | 误差 $\|V_k-V^*\|_\infty$ |
| --- | -------- | -------- | ----------------------- |
| 0   | 0        | 0        | $8/3$                   |
| 1   | 2        | 0        | $4/3$                   |
| 2   | 2        | 1        | $2/3$                   |
| 3   | 2.5      | 1        | $1/3$                   |
| 4   | 2.5      | 1.25     | $1/6$                   |

其中无穷范数是各个绝对值中的最大值。
(c)
对于相邻两步之间的误差比 $\frac{\|V_{k+1}-V^*\|_\infty}{\|V_k-V^*\|_\infty}$ ，我们发现它在上表中严格为 $0.5$ 。
这就是 $\gamma$-收缩的字面含义，每次迭代误差收缩到原来的 $\gamma$ 倍。
(d) 初始误差 $\|V_0-V^*\|_\infty$ 一定满足小于等于 $R_{\max}/(1-\gamma)$ 。

这是因为：设 $|r| \le R_{\max}$ ，有折扣回报的上界
$$|G_t|=|r_t+\gamma r_{t+1}+\gamma^2 r_{t+2} + \dots| \le R_{\max} + \gamma R_{\max}+\gamma^2R_{\max}+\dots = R_{\max}/(1-\gamma)$$
值函数是折扣回报的期望，则既然每个折扣回报都小于等于这个上界，它们的期望也一定有
$$|V^*(s)| \le R_{\max}/(1-\gamma)$$
设初始值函数 $V_0$ 为全零向量，则有
$$\|V_0-V^*\|_\infty \le R_{\max}/(1-\gamma)$$
得证。

按照压缩映射，
$$\|V_k-V^*\|_\infty \le \gamma\|V_{k-1}-V^*\|_\infty \le \dots \le \gamma^k \|V_{0}-V^*\|_\infty$$
假设 $\gamma=0.9$ ，假设 $R_{\max}=1$（其实对于这道题的两个即时奖励应该是 $\max \{2,0\}=2$ ，但不重要，能够说明问题就行），则有初始误差 $\le R_{\max}/(1-\gamma)=10$ 。
要使值迭代的误差 $\le 10^{-2}$ ，则有
$$0.9^k \cdot 10 \le 10^{-2}$$
解得$$k \ge \frac{-3 \ln 10}{\ln 0.9} \approx 65.6$$
故至少需要66次迭代，才能把误差降到0.01以下。
(e) $\gamma$ 的值决定了我们的“有效视野”，即对长期回报的关注度。
定义 $H_{eff}=1/(1-\gamma)$ ，则对于各个 $\gamma$ 值有：

| $\gamma$ | $H_{eff}$ |
| -------- | --------- |
| 0.9      | 10        |
| 0.99     | 100       |
| 0.999    | 1000      |

$\gamma$ 每靠近1一个数量级，视野放大10倍。
这个“有效视野”衡量智能体做决策时，大致“往前看多少步”。常用的一种说法是当 $γ^t≈e^{-1}≈0.368$ 时，对应的 $t$ 可看作“有效视野”。
近似 $\ln t \approx t-1$ ，则 $\gamma^t=e^{-1}$ 正对应上述的 $t \approx 1/(1-\gamma)$ 。于是 $H_{eff}$ 对应的是折扣系数衰减到约37%所需的步数。
## 1 Bellman方程

考虑一个3状态MDP，$\gamma=0.5$ ，状态 $\{1,2,3\}$ ，每个状态只有一个动作（无控制，直接评估）：
- 状态1：奖励 $R_1=2$ ，确定转到状态 $2$ ，
- 状态2：奖励 $R_2=0$ ，确定转到状态 $3$ ，
- 状态3：奖励 $R_3=4$ ，确定转到状态 $1$ 。
(a) 写出三条Bellman期望方程。
$$V(1)=R_1+\gamma V(2)=2+0.5V(2)$$
$$V(2)=R_2+\gamma V(3)=0+0.5V(3)$$
$$V(3)=R_3+\gamma V(1)=4+0.5V(1)$$
(b) 写成矩阵形式有
$$V=R+\gamma PV$$
其中 $R = (2,0,4)^T \in \mathbb{R}^3$ ；$P \in \mathbb{R}^{3 \times 3}$ 满足 $P_{1,2}=1, P_{2,3}=1, P_{3,1}=1$ ，其余项为0。

$P$ 的形式可以通过以下思考得到：
由于
$$(PV)_1=P_{1,1}V_1+P_{1,2}V_2+P_{1,3}V_3$$
（直觉上思考一个 $3 \times 3$ 的矩阵乘以一个 $\mathbb{R}^3$ 的向量，取前者的行、和后者进行点积）
我们为了让这个乘出来的结果为 $V_2=V(2)$ ，则应该只取中间那个项（系数为1），其余为0。

设 $V=(V(1),V(2),V(3))^T$ ，此时有 $PV=(V(2),V(3),V(1))^T$ 。
(c) 解线性方程组有
$$(I-\gamma P)V=R$$
解得 $V(1)={24 \over 7}, V(2)={20 \over 7}, V(3)={40 \over 7}$ 。
(d) 直接展开求前6项的部分和有
$$V(1)=2+0.5 \times 0+0.25 \times 4 + 0.125 \times 2+0.0625 \times 0 + 0.03125 \times 4=3.375$$
又 $\frac{24}{7} \approx 3.428571$ ，误差约为 $0.053571$ ，小于误差上界 $\gamma^6 \cdot R_{max}/(1-\gamma)=0.125$ 。

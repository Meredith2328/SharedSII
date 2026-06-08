> edit time: 2026-06-08 00:22:44 PPO，RLHF的PPO，DPO
> edit time: 2026-06-08 12:20:33 IPO，KTO，GRPO，收敛性分析

所有主流PO方法都是在优化KL正则化的目标，只是 $\text{Quality}$ 的定义不同：
$$
\max_{\pi} \mathbb{E}[\text{Quality}(\pi)] - \beta D_{KL}(\pi \| \pi_{\text{ref}})
$$

| 方法       | $\text{Quality}$ 定义                                | 数据类型 |
| -------- | -------------------------------------------------- | ---- |
| RLHF-PPO | $\mathbb{E}_{y \sim \pi}[R(y)]$                    | 奖励模型 |
| DPO      | $\mathbb{E}[\log P(y_w \succ y_l)]$（隐式）            | 成对偏好 |
| IPO      | $\mathbb{E}[(R(y_w) - R(y_l) - m)^2]^{-1}$（隐式）     | 成对偏好 |
| KTO      | $\mathbb{E}[\text{sign}(y) \cdot \log \pi(y)]$（近似） | 单点标注 |
| GRPO     | $\mathbb{E}[\text{rank}(y) \cdot \log \pi(y)]$（近似） | 群体采样 |
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
顺便补充一些Deepseek V4论文中的讨论。
GRPO只是组内平均——就像是一个模型有多个差的回答，能给那个稍微好一点的回答正反馈——但是PPO有一个Value Model，是不是PPO的上限更高？
以及，GRPO更往前的路在哪里？

我个人没有找到很好的讨论。根据个人的直觉，打分模型更可能左脚踩右脚上天；可是GRPO这样更简洁的构造，却更可能稳定、充分地发挥架构的优势。也许这是一个权衡。

但有些事情我们是能够找到证据的。根据论文第5.1节说明：
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

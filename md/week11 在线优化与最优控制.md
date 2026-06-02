> 玩具：泛函 $J[y]$ （$y=y(x)$），变分 $\frac{\delta J}{\delta y}$ ，熵 $H[p]=-\int p\log p dy$ ，KL散度，Bandit……
> edit time: 2026-05-15 20:12:10
## 目录

Regret
Hoeffding：随着n增大，不喜欢的概率是指数级下降的（$e^n$）。
Bandit：最小化累积Regret
- $\epsilon-Greedy$，Regret线性增加，如果逐渐降低 $\epsilon$ 的话不好做，是很技术的问题
- UCB，对于不确定性乐观，无悔（平均界 $\rightarrow 0$，最差界 $\rightarrow \sqrt{T}$）
- Thompson Sampling（与UCB同阶）
最优学习：已经找到一个最大值之后，通过积极地尝试其他选项，降低其他选项可能成为最大值的概率
- KG（Knowledge Gradient）

泛函
Euler-Lagrange方程
最大熵原理
KL散度和变分推断：用q 编码来自p 的数据，平均多花多少bits
最小化KL散度等价于最大化ELBO
从最优控制到Bellman方程

exercise1-exercise10
## 变分法与最优控制知识部分
****
Euler-Lagrange方程：设函数 $y(x)$ ，泛函
$$J[y]=\int F(y,y',x)dx$$
取极值时必有
$$\frac{\partial F}{\partial y}-\frac{d}{dx} \frac{\partial F}{\partial y'}=0$$
****
最大熵原理：选择满足约束的熵最大的分布。
$$\max_p H[p]=-\int p(x)\log p(x)dx$$
subject to 约束条件
另外，熵的形式是 $-\int p\log pdx$ ，也可以把 $p$ 看作概率密度，写作 $H[p]=-\mathbb{E}_p [\log p]$ ，即 $p$ 的熵等价于负的 $\log p$ 关于 $p$ 的期望。  
关于泛函求变分，有关键公式：
$$\frac{\partial}{\partial p(x)}\int f(p(y))dy=f'(p(x))$$
这个公式形式有点像变上限的积分求导公式。可以推得以下二级结论：
$$\frac{\partial}{\partial p(x)} \int p(y)dy=1$$
$$\frac{\partial}{\partial p(x)}\int p \log p dy=\log p+1$$
以及对指数分布，有
$$H[p]=\ln \mu+1$$
无约束时，均匀分布熵最大。
给定均值时，指数分布熵最大。
给定均值和方差时，高斯分布熵最大。
一般地，最大熵分布是指数族分布的形式。
****
KL散度：用q编码来自p的数据，平均多花多少bits。
$$D_{KL}(p||q)=\int p(x) \log \frac{p(x)}{q(x)}dx=\mathbb{E}_p[\log \frac{p(x)}{q(x)}]$$
注意KL散度具有不对称性（见eg3），被使用的写在后面和分母处。

变分推断是希望找简单分布 $q(z)$ 近似使得 $q(z) \approx p(z|x)$ ，从而使得贝叶斯推断问题可求。
于是考虑使用 $D_{KL}(p||q)$ 或者 $D_{KL}(q||p)$ ，前者在计算上不可行（同贝叶斯推断），无奈之下只能使用后者。
容易推导得到，（当然，$\log p(x)$ 也可以写作 $E_q[\log q(z)]$）
$$\log p(x)=D_{KL}(q||p)+\int q \log \frac{p(x,z)}{q(z)}dz=D_{KL}(q||p)+ELBO(q)$$
考虑到 $D_{KL}(q||p) \ge 0$ ，有 $\log p(x) \ge ELBO(q)$ 。故称 $ELBO$ 为evidence lower bound。
最小化KL等价于最大化ELBO，即给定 $q$ ，最大化 $\log p(x,z)$ 和 $\log q(z)$ 的期望之差。
$$ELBO(q)=\mathbb{E}_q[\log p(x,z)]-\mathbb{E}_q[\log q(z)]$$
当然，也可以写成下述两种等价形式：
（联合对数似然的期望 + q的熵；重构项 - 正则项）
$$ELBO(q)=\mathbb{E}_q[\log p(x,z)]+H[q]$$
$$ELBO(q)=\mathbb{E}_q[\log p(x|z)]-D_{KL}(q(z)||p(z))$$
变分推断就是为了通过最大化下界ELBO，来间接优化难以直接计算的对数证据 $\log p(x)$ 。
****
LQR问题，我们从一个数值例子反着来。
LQR的线性动力学：$\dot{x}=Ax+Bu$
LQR的二次代价：
$$J=\frac{1}{2}x(T)^TQ_fx(T)+\frac{1}{2}\int_0^T(x^TQx+u^TRu)dt$$
其中 $f(x,u)=Ax+Bu$ ，$L(x,u)=x^TQx+u^TRu$ 。
数值例子：
$\dot x=x+u, J=\frac{1}{2}x(T)^2+\frac{1}{2}\int_0^T(x^2+u^2)dt$ ，目标是控制 $x$ 到 $0$ ，同时不用太大的 $u$ 。
对于该例子，$Q=R=Q_f=A=B=1$ ，
Ricacci方程为（LQR的Ricacci方程形式为 $-\dot{P}=Q+A^TP+PA-PBR^{-1}B^TP$）
$$-\dot{P}=1+2P-P^2$$
稳态令 $\dot{P}=0$ 有 $P=1+\sqrt{2}$ ，
最优控制 $u^*=-Px \approx -2.414x$ （LQR的最优控制为 $u^*=-R^{-1}B^TPx$），
闭环动力学 $\dot{x} \approx -1.414x$ ，稳定，极点在$-1.414$ 。
如果不加控制（令 $u=0$），则有 $\dot{x}=x$ ，这一一阶微分形式使得指数增长，不稳定。

接下来反过来看怎么推出LQR的Ricacci方程。
HJB方程（Hamilton-Jacobi-Bellman方程）：
$$-\frac{\partial V}{\partial t}=\min_u [L(x,u,t)+\nabla_x V \cdot f(x,u,t)]$$
猜测LQR的 $V(x,t)=\frac{1}{2}x^TP(t)x$ ，
则有 $\nabla_xV=Px$ ，$\frac{\partial V}{\partial t}=\frac{1}{2}x^T\dot{P}x$ ，
代入HJB方程，将HJB方程视为 $u$ 的函数，最小化 $u$ 得最优控制
$$u^*=-R^{-1}B^TPx$$
这是一个对 $x$ 的线性反馈。将其代回HJB方程就得到了LQR的Ricacci方程
$$-\dot{P}=Q+A^TP+PA-PBR^{-1}B^TP$$
从而前述的数值例子是从HJB这个根本的方程推得的。（HJB由Bellman最优性原理推得）
****
## 在线优化与Regret框架知识部分

****
$$
\text{Regret}_T = \underbrace{\sum_{t=1}^T \ell_t(x_t)}_{\text{你的总损失}} - \underbrace{\min_{x \in \mathcal{X}} \sum_{t=1}^T \ell_t(x)}_{\text{最佳固定策略}}
$$
注意，总损失是由选择 $x_1,x_2, \dots, x_T$ 得到的，
而最佳固定策略的选择在各轮都是相同固定的最佳选择 $x$ 。
这意味着，我们的动态切换可能会比最佳固定策略好，但也可能因为动态切换而损失更高。

**Regret**的量级分为 $O(T)$ ，$O(\sqrt{T})$ 和 $O(\log T)$ ，其中 $O(\sqrt{T})$ 开始是平均后悔趋于0的（$T \rightarrow \infty, O(\frac{\sqrt{T}}{T}) \rightarrow 0$），符合我们的需求。$O(\log T)$ 也可能达到，但需要更强的假设。
****
**MWU**：给每个专家打分，表现差的惩罚，惩罚是乘性的（指数惩罚）。
1: 每个专家初始权重 $w_i = 1$  
2: 选一个学习率 $\eta$ (比如 0.1)  
3: for 每一轮 $t = 1, 2, \dots, T$ do  
4:    计算概率：$p_i = \frac{w_i}{\sum_j w_j}$（权重归一化）  
5:    按概率随机选一个专家  
6:    看到所有专家的损失 $\ell_t(i)$ (0 到 1 之间)  
7:    乘法更新：$w_i \leftarrow w_i \times (1 - \eta \cdot \ell_t(i))$  
8: end for
****
**Hoeffding不等式**：采样n次，偏差大的概率指数级衰减。采样越多，估计越准。
定理 4.1 (Hoeffding 不等式). $n$ 个独立样本 $X_i \in [0,1]$，样本均值 $\bar{X} = \frac{1}{n} \sum X_i$，真实均值 $\mu$，则：
$$
P(|\bar{X} - \mu| \geq \epsilon) \leq 2e^{-2n\epsilon^2}
$$
即样本均值偏移真实均值，偏差更大的概率随着样本数 $n$ 指数级衰减。
比如抛硬币，想知道估计误差超过 $\epsilon=0.1$ 的概率，
$n=100$ 时， $p(|\hat{p}-p| \ge 0.1) \le 0.27$ ，即还有27%的概率估计误差超过0.1。
$n=500$ 时，$p(|\hat{p}-p| \ge 0.1) \le 0.00009$ ，即还有0.009%的概率估计误差超过0.1，基本不可能。
如果想要95%的把握误差不超过 $\epsilon$ ，则 $\epsilon$ 应通过以下式子求得：
$$2e^{-2n \epsilon^2}=0.05$$
可见，置信区间宽度正比于 $1/\sqrt{n}$ 。
这意味着，想把误差减半，需要4倍样本；想把误差减到十分之一，需要100倍的样本。
这与Regret为 $O(\sqrt{T})$ 不谋而合：我们获取信息的速度就是 $\sqrt{T}$ 。
****
**Bandit**：有多台老虎机，各自收益不同。每次选一台拉，拉 $T$ 次，总收益最大。
（平衡探索与利用）
**$\epsilon$-Greedy**：
以 $1-\epsilon$ 的概率选目前估计最好的，以 $\epsilon$ 概率随机选一个。
**UCB**：
$$UCB_a(t)=\hat{\mu_a}+\sqrt{2 \ln t \over n_a}$$
第二项 $\sqrt{2\ln t \over n_a}$ 是探索奖励，因为拉的次数较少时，该项较大，奖励拉该臂更多次。
**Thompson Sampling**：将每个臂先验置为 $Beta(1,1)$ ，采样各个分布，选取最大值，看结果如何，并更新相应臂的分布。
1: 每台机器初始化：$\alpha_a = 1, \beta_a = 1$ (先验置为 $Beta(1,1)$)
2: for 每一轮 do
3:    对每台机器 $a$，从 $Beta(\alpha_a, \beta_a)$ 采样一个 $\theta_a$
4:    选择 $\theta$ 最大的机器
5:    观察结果：赢则 $\alpha_a \leftarrow \alpha_a + 1$，输则 $\beta_a \leftarrow \beta_a + 1$
6: end for
****
**最优学习**：最大化最终选择的 $\mu$ 值。“更激进的探索最优”
**Knowledge Gradient**（KG）：应选择KG值最大的臂进行探索（“一次实验改进的决策”）。
$$KG(a)=\tilde{\sigma_a} \cdot f(z)$$
其中有效标准差为后验标准差和噪声标准差求得的
$$\tilde{\sigma_a}=\frac{\sigma_a^2}{\sqrt{\sigma_a^2+\sigma_\epsilon^2}}$$
$z$ 为当前均值和“对手最佳” $\nu_{-a}=\max_{a' \ne a} \hat{\mu_{a'}}$ 的差值绝对值再归一化
$$z=\frac{|\hat{\mu_a}-\nu_{-a}|}{\tilde{\sigma_a}}$$
$f$ 为标准正态分布期望
$$f(z)=z \Phi(z)+\phi(z),z \ge 0$$

## eg1. Euler-Lagrange方程基础
$$F=(y')^2+y^2$$
(a) Euler-Lagrange方程有
$$2y-\frac{d (2y')}{dx}=0$$
即
$$y''-y=0$$
(b) 二阶常微分方程。
$$r^2-1=0$$
解得 $r_1=1,r_2=-1$ 。
故有
$$y(x)=C_1e^x+C_2e^{-x}$$
代入 $y(0)=0,y(1)=1$ 得
$$C_1=\frac{1}{e-e^{-1}},C_2=-C_1,y(x)=\frac{e^x-e^{-x}}{e-e^{-1}}$$
又有 $e^x-e^{-x}=2 \sinh(x)$ ，故有我们要找的满足Euler-Lagrange方程的特定的极值函数
$$y^*(x)=y(x)=\frac{\sinh x}{\sinh 1}$$
(c) 由(b)得
$$y'(x)=\frac{\cosh x}{\sinh 1}$$
注意到
$$\frac{d}{dx}(yy')=(y')^2+yy''$$
又有 $y''=y$ ，则有
$$\frac{d}{dx}(yy')=y^2+(y')^2=F$$
故
$$J[y]=\int_0^1 Fdx=\int_0^1 \frac{d}{dx}(yy')dx$$
$$J[y]=yy'|_0^1$$
考虑到 $\sinh 0=0, \cosh 0=1$ ，
当 $x=0$ 时，$y'(0)=\frac{1}{\sinh 1}$ ，$y(0)=0$ ，$yy'=0$ ，
当 $x=1$ 时，$y'(1)=\frac{\cosh 1}{\sinh 1}$ ，$y(1)=1$ ，$yy'=\frac{\cosh 1}{\sinh 1}$ 。
故有泛函最小值
$$J^*=\frac{\cosh 1}{\sinh 1}=\coth 1 \approx 1.313$$
(d) 当 $y=x$ 时，$y'=1$ 。
$$J[y]=\int_0^1 (1+x^2)dx=(x+\frac{1}{3}x^3)|_0^1=\frac{4}{3}>J^*$$
即不满足E-L方程的试探函数 $y=x$ 的 $J$ 值大于 $J^*$ ，不为极值。
## eg2. 约束下的最大熵分布

设 $x \in [0,\infty)$ ，已知 $\mathbb{E}[x]=3$ 。
(a) 由 $\mathbb{E}[x]=3$ 可得问题约束条件为归一化和均值约束
$$\int_0^\infty p(x)dx=1$$
$$\int_0^\infty xp(x)dx=3$$
使用Lagrange乘子法求最大熵，设
$$L=-\int p \log p dx-\lambda_0 (\int p(x)dx-1)- \lambda_1(\int xp(x)dx-3)$$
变分有
$$\frac{\partial L}{\partial p}=-\log p-1-\lambda_0-\lambda_1x=0$$
(b) 解出最大熵分布 $p^*(x)$ 的解析形式有
$$p=e^{-1-\lambda_0-\lambda_1x}=Ce^{-\lambda_1x}$$
(c) 满足指数分布的形式：$p(x)=\frac{1}{\mu} e^{-x/\mu}$ 。
故给定 $\mu=3$ 则有
$$p(x)=\frac{1}{3}e^{-\frac{x}{3}}$$
(d) 
$$\ln p=\ln \frac{1}{3}-\frac{x}{3}=-\ln 3-\frac{x}{3}$$
微分熵 $H[p^*]$ 为
$$H[p^*]=-\int_0^\infty p \ln p dx$$
暂时不代入 $\mu=3$ ，考虑指数分布的微分熵的一般情况。有
$$H[p^*]=-\int p(-\ln \mu-\frac{x}{\mu} )dx=\ln \mu+\frac{1}{\mu}\int pxdx=\ln \mu+\frac{1}{\mu} \cdot \mu=\ln \mu +1$$
故有本问题的 $H=1+\ln 3 \approx 2.099 \text{ nats}$ 。

## eg3. 高斯分布之间的KL散度

(a)
$$D_{KL}(N(\mu_1,\sigma_1^2)||N(\mu_2,\sigma_2^2))=\int p_1 \log\frac{p_1}{p_2} dx$$
解析解为
$$D_{KL}(N_1||N_2)=\log \frac{\sigma_2}{\sigma_1}+\frac{\sigma_1^2+(\mu_1-\mu_2)^2}{2\sigma_2^2}-\frac{1}{2}$$
(b)
代入 $\mu_p=0,\sigma_p=1,\mu_q=1,\sigma_q=2$ 得
$$D_{KL}(p||q)=\ln 2+0.25-0.5 \approx 0.443$$
(c)
$$D_{KL}(q||p)=-\ln 2+2.5-0.5 \approx 1.307$$
(d)
$$D_{KL}(p||q)<D_{KL}(q||p)$$
$D_{KL}(p||q)$ 鼓励 $q$ 覆盖 $p$ 的高概率区域（zero-forcing，$q$ 在 $p$ 概率大的地方也要大，可以忽略 $p$ 的尾部），
而 $D_{KL}(q||p)$ 鼓励 $q$ 平均拟合 $p$（mean-seeking，$q$ 在 $p$ 概率小的地方也要小，否则会受很大惩罚，从而让 $q$ 尽量覆盖 $p$ 的全部支撑集）。

## eg4. VAE的KL正则项

(a)
$$D_{KL}(q||p)=\frac{1}{2}\sum_{j=1}^d(\mu_j^2+\sigma_j^2-1-\log \sigma_j^2)$$
(b) $d=3$ ，有三项。
$$D_{KL}(q||p)=\frac{1}{2} [(0.5^2+0.5-1-\log 0.5)+((-0.5)^2+1.5-1-\log 1.5)+(1^2+0.4-1-\log 0.4)]$$
$$=\frac{1}{2}[0.4431471806+0.3445348919+1.093147181]$$
第一个分量约为 $0.222$ ，第二个分量约为 $0.172$ ，第三个分量约为 $0.658$ 。
(c)
$$D_{KL}(q||p) \approx 0.222+0.172+0.658=1.052$$
(d)
第三个分量（$\mu=1$，$\sigma^2=0.4$）贡献最大。因为它均值偏离 $(0, 1)$ 程度最大（过大），$\sigma^2$ 偏移的程度也较大（过小），KL散度对均值和方差的偏移均有正惩罚。

## eg5. LQR与代数Riccati方程

(a)
$$-\dot{P}=Q+A^TP+PA-PBR^{-1}B^TP$$
代入 $A=-1, B=2, Q=8, R=2$ 得
$$-\dot{P}=8-2P-2P^2$$
令 $\dot{P}=0$ 得
$$P^2+P-4=0$$
(b)
$$P_1=\frac{-1 +\sqrt{17}}{2},P_2=\frac{-1 -\sqrt{17}}{2}$$
稳态正解 $P_1 \approx 1.562$ 。
(c)
$$u^*=-R^{-1}B^TPx=-Px$$
故反馈增益 $K=-P_1=-1.562$ ，$u^*=-1.562x$。
(d)
$$\dot{x}=-x+2u=-x-2 \times 1.562x=-4.124x$$
实部为负，系统稳定。
(e)
原系统 $\dot{x}=-x$ 本就稳定，闭环极点把收敛速度提速了 $4.124$ 倍。
## eg6. MWU手算三轮

(a) 每轮概率分布与期望损失
第一轮
$$p^{(1)}=(1/3,1/3,1/3)$$
$$L_1=1/3 \times 0.5+1/3 \times 0.2+1/3 \times 0.8=0.5$$
$$w^{(2)}_i=w^{(1)}_i(1-\eta l_1(i))$$
则有
$$w^{(2)}=(1 \times (1-0.4 \times 0.5), 1 \times (1-0.4 \times 0.2), 1 \times (1 - 0.4 \times 0.8))=(0.8,0.92,0.68)$$
第二轮 $0.8+0.92+0.68=2.4$
$$p^{(2)}=(0.8/2.4, 0.92/2.4,0.68/2.4)$$
$$L_2=0.8/2.4 \times 0.3 + 0.92/2.4 \times 0.7+0.68/2.4 \times 0.4 \approx 0.482$$
$$w^{(3)}=(0.8 \times (1-0.4 \times 0.3), 0.92 \times (1-0.4 \times 0.7), 0.68 \times (1-0.4 \times 0.4))$$
$$w^{(3)}=(0.704, 0.6624, 0.5712)$$
第三轮 $0.704+0.6624+0.5712=1.9376$
$$p^{(3)} \approx (0.3633,0.3419,0.2948)$$
$$L_3 \approx 0.3501$$
综上有：（以下均为约数）
$$p^{(1)} = (0.3333,0.3333,0.3333),L_1=0.5$$
$$p^{(2)} = (0.3333,0.3833,0.2833),L_2=0.4816$$
$$p^{(3)} = (0.3633,0.3419,0.2948),L_3=0.3501$$
(b) 每轮更新后的权重
$$w^{(2)}=(0.8,0.92,0.68)$$
$$w^{(3)}=(0.704,0.6624,0.5712)$$
$$w^{(4)}=(0.53504,0.582912,0.548352)$$
(c) 总期望损失
$$\sum_{t=1}^3 L_t=0.5+0.4816+0.3501=1.3317$$
(d) 最优固定专家：
固定专家1总损失 $0.5+0.3+0.6=1.4$
固定专家2总损失 $0.2+0.7+0.3=1.2$
固定专家3总损失 $0.8+0.4+0.1=1.3$
故最优固定专家 $i^*=2$ ，其总损失为 $1.2$ 。
(e)
Regret为 $1.3317-1.2=0.1317$ 。
MWU的理论上界为
$$\frac{\ln N}{\eta}+\eta T=\frac{\ln 3}{0.4}+0.4 \times 3 \approx 3.9465$$
实际Regret远小于上界。
## eg7. Hoeffding不等式与样本量

(a) 要求95%把握下误差不超过 $\epsilon$ ，则最少样本量 $n$ 关于 $\epsilon$ 的表达式
令
$$2e^{-2n \epsilon^2}=0.05$$
有
$$n=-\frac{\ln 0.025}{2\epsilon^2}$$
(b) 当 $\epsilon=0.02$ 时，
$$n=-\frac{\ln 0.025}{2 \times 0.02^2}=4611.099318$$
95%把握下误差不超过 $0.02$ ，至少需要 $4612$ 个样本。
(c) 当 $\epsilon=0.01$ 时，
$$n=-\frac{\ln 0.025}{2 \times 0.01^2}=18444.39727$$
95%把握下误差不超过 $0.01$ ，至少需要 $18445$ 个样本。这是 (b) 的约4倍。$\epsilon$ 减半，$1/\epsilon^2$ 变为原来的4倍，与样本量的变化相同。即所需的样本量正比于 $1/\epsilon^2$ 。
(d) 取 $\epsilon=0.05$ ，
$$P(|\overline{X}-p| \ge 0.05) \le 2 \cdot e^{-2 \times 200 \times 0.05^2}=2e^{-1} \approx 0.736$$
则做200次实验，误差小于等于0.05的把握（概率下界）是 $0.264$ 。
## eg8. UCB决策

(a)
$$UCB_a(t)=\hat{\mu_a}+\sqrt{2 \ln t \over n_a}$$
第二项 $\sqrt{2\ln t \over n_a}$ 是探索奖励，因为拉的次数较少时，该项较大，奖励拉该臂更多次。
(b)
$$UCB_1(14)=0.4+\sqrt{\frac{2 \ln 14}{5}} \approx 1.427$$
$$UCB_2(14)=0.6+\sqrt{\frac{2 \ln 14}{3}} \approx 1.927$$
$$UCB_3(14)=0.5+\sqrt{\frac{2 \ln 14}{2}} \approx 2.124$$
$$UCB_4(14)=0.55+\sqrt{\frac{2 \ln 14}{4}} \approx 1.699$$
(c)
第15轮的选择是arm 3。
因为要充分探索，而arm 3被拉的次数较少（仅2次），比经验均值最大的arm 2要少。
(d)
$$Regret_T \le \frac{8 \ln 10000}{0.3} + \frac{8 \ln 10000}{0.2} + \frac{8 \ln 10000}{0.1} \approx 1350.85$$
## eg9. Thompson Sampling

两个伯努利臂，先验为 $Beta(1,1)$ ，
观测到arm A有8次成功、2次失败；arm B有3次成功、7次失败。
(a)
arm A的后验分布为 $Beta(9,3)$ ，arm B的后验分布为 $Beta(4,8)$ 。
arm A：
均值为 9/(9+3)=0.75 ，
标准差为 √(9\*3/((9+3)^2\*(9+3+1)))=√(27/(144\*13))≈√0.0144=0.12。
arm B：
均值为 4/(4+8)=0.3333333333，
标准差为 √(4\*8/((4+8)^2\*(4+8+1)))=√(32/(144\*13))≈√0.017094≈0.13。
(b)
假设 $\theta_A \sim N(\mu_1,\sigma_1^2)$ ，$\theta_B \sim N(\mu_2,\sigma_2^2)$ 。则
$$\theta_A-\theta_B \sim N(\mu_1-\mu_2,\sigma_A^2+\sigma_B^2)$$
$$\mu_A-\mu_B=0.75-0.3333=0.4167$$
$$\sigma_A^2+\sigma_B^2=0.0144+0.0171=0.0315$$
$$\sqrt{\sigma_A^2+\sigma_B^2}=\sqrt{0.0315} \approx 0.1775$$
$$Pr(\theta_A > \theta_B)=Pr(\theta_A-\theta_B>0)=\Phi(\frac{0.4167}{0.1775})=\Phi(2.348)$$
给出的标准正态值有 $\Phi(2.34) \approx 0.990$ ，可以近似所求也为 $0.990$ 。
(c)
$$\alpha_A'=10,\beta_A'=3,\mu'_A=\frac{10}{13} \approx 0.7692, \sigma_A' \approx 0.1126$$
Arm B不变，仍为 $\mu_b \approx 0.3333,\sigma_B=0.1307$ 。
故
$$\mu_A'-\mu_B \approx 0.4359, \sqrt{\sigma_A'+\sigma_B^2} \approx 0.1725, z=\frac{0.4359}{0.1725} \approx 2.527$$
$$Pr(\theta_A > \theta_B) \approx \Phi(2.527) \approx 0.994$$
新观测后，概率从0.991提升到约0.994，即选择arm A的信心更强。
(d)
差距不变但样本量增加时，后验分布更集中，$Pr(\theta_A > \theta_B)$ 会增大。
## eg10. Knowledge Gradient与最优学习

(a)
$$\tilde{\sigma}_1=\frac{\sigma_1^2}{\sqrt{\sigma_1^2+\sigma_\epsilon^2}}=\frac{0.05^2}{\sqrt{0.05^2+0.1^2}}=0.02236$$
$$\nu_{-1}=\max\{0.65,0.5,0.68\}=0.68$$
$$\tilde{\sigma}_2=\frac{\sigma_2^2}{\sqrt{\sigma_2^2+\sigma_\epsilon^2}}=\frac{0.2^2}{\sqrt{0.2^2+0.1^2}}=0.17889$$
$$\nu_{-2}=\max\{0.7,0.5,0.68\}=0.7$$
同理有 $\tilde{\sigma_3}=0.28461$ ，$\nu_{-3}=0.7$ ，$\tilde{\sigma_4}=0.07071$，$\nu_{-4}=0.7$ 。
(b)
对于Arm 1，
$$z_0={0.02 \over 0.02236} \approx 0.8944$$
对于Arm 2，
$$z_0={0.05 \over 0.17889} \approx 0.2795$$
对于Arm 3，
$$z_0={0.2 \over 0.28461} \approx 0.7026$$
对于Arm 4，
$$z_0={0.02 \over 0.07071} \approx 0.2828$$
则
$$KG(1)=0.0023,KG(2)=0.0490,KG(3)=0.0401,KG(4)=0.0193$$
(c)
最大的为 $KG(2)$ ，故选择arm 2。
(d)
如果立即commit，会选择当前均值最大的arm 1，即
$$\max_a \hat{\mu_a}=0.70$$
实验的KG=0.0490，这是再做一次实验可能带来的信息价值
（期望改进量，注意不是概率）。
(e)

| arm $a$ | $\hat{\mu}$ | $\sigma_a$ | $KG$   | $\tilde{\sigma_a}$ | $z_0^{(a)}$ |
| ------- | ----------- | ---------- | ------ | ------------------ | ----------- |
| 1       | 0.70        | 0.05       | 0.0023 | 0.0224             | 0.8944      |
| 2       | 0.65        | 0.20       | 0.0490 | 0.1789             | 0.2795      |
| 3       | 0.50        | 0.30       | 0.0401 | 0.2846             | 0.7026      |
| 4       | 0.68        | 0.10       | 0.0193 | 0.0707             | 0.2828      |

对比Arm 1和Arm 2，发现Arm 2的 $\hat{\mu}$ 更低，但是 $KG$ 反而更大。
因为Arm 2的有效后验更新标准差 $\tilde{\sigma_a}$ 明显更大，且 $z_0^{(a)}$ 更小、使得 $f(z_0)$ 更大，$KG$ 是这两项的乘积，所以Arm 2的KG明显比Arm 1大。

(f)
$$\tilde{\sigma_3}=\frac{0.09}{\sqrt{0.09+0.25}} \approx 0.1543$$
$$\nu_{-3}=0.70$$
$$z_0 \approx 1.296$$
$$f(z_0) \approx 0.046$$
$$KG(3) \approx 0.0071$$
比原来的 $0.0401$ 显著下降。这是因为单次实验的有效后验更新标准差 $\tilde{\sigma_a}$ 显著变小（受限于噪声标准差 $\sigma_\epsilon$ ），即使 $z_0$ 可能变大，但是 $KG$ 整体下降，使得信息价值变小。

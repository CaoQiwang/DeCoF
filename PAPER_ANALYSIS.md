# Autoformer、FEDformer、STWave 与 STDN 方法分析

本文基于以下四篇本地论文，重点分析它们如何分解时序、如何建模趋势与高频/季节动态、如何融合不同成分，以及这些设计对当前 DeCoF 架构的启发。

- [AutoFormer.pdf](../AutoFormer.pdf)
- [FEDformer.pdf](../FEDformer.pdf)
- [STWave.pdf](../When_Spatio-Temporal_Meet_Wavelets_Disentangled_Traffic_Forecasting_via_Efficient_Spectral_Graph_Attention_Networks.pdf)
- [STDN.pdf](../STDN.pdf)

> 页码均指本地 PDF 页码。Autoformer 重点参见第 3-5 页，FEDformer 重点参见第 3-6 页，STWave 重点参见第 4-7 页，STDN 重点参见第 3-5 页。

## 1. 总体结论

四篇论文都在做“复杂序列分解后分别建模”，但分解含义并不相同：

| 模型 | 分解发生在哪里 | 粗粒度成分 | 细粒度成分 | 是否严格频率分解 | 是否使用专门的双分支网络 |
| --- | --- | --- | --- | --- | --- |
| Autoformer | 时间域移动平均 | Trend-Cyclical | Seasonal Residual | 否 | 否，季节项走主干，趋势项逐层累积 |
| FEDformer | 多尺度移动平均 + 频域表示学习 | Trend | Seasonal Residual | 部分；分解本身仍是移动平均 | 否，仍是季节主干与趋势累积 |
| STWave | 多层离散小波变换 DWT | Low-frequency Trend | Multi-level High-frequency Event | 是 | 是，趋势和事件使用不同时间模块 |
| STDN | 隐空间时空门控 | Spatio-temporal Trend | Seasonal Residual | 否 | 是，两部分分别经过 GRU 编码器 |

对当前 DeCoF 最重要的结论有四点：

1. **高频分支不一定需要同时包含 $H_r$ 和 $H_d$。** Autoformer、FEDformer 和 STDN 都证明了“从粗成分中相减得到的残差”可以独立作为细粒度成分；STWave 则只使用小波高频项作为事件分支。
2. **低秩表示不自动等于低频表示。** Autoformer 使用移动平均，STWave 使用 DWT，STDN 使用显式时空门控。DeCoF 如果只用低秩注意力产生 $H_c$，需要额外证据说明 $H_c$ 确实是粗粒度或低频成分。
3. **不同成分最好使用不同归纳偏置。** STWave 对低频趋势使用全局时间注意力，对高频事件使用局部因果卷积，这是四篇论文中粗细分工最明确的设计。
4. **直接相加不是唯一也未必是最佳融合。** STWave 专门设计了自适应事件融合；其消融实验中，用直接加法替换自适应融合会导致性能下降。

## 2. Autoformer

论文：*Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting*。

### 2.1 核心问题

Autoformer 面向长序列预测，主要解决：

- 长期序列包含趋势、周期和局部变化，直接用 Transformer 很难建模；
- 点级 Self-Attention 不能充分利用周期子序列之间的相似性；
- 长序列注意力具有较高的计算开销。

它的两个核心设计是：

1. 将序列分解作为网络内部的基本操作，而不是仅在输入前预处理一次；
2. 用 Auto-Correlation 代替 Self-Attention，寻找周期对应的时间延迟并聚合相似子序列。

### 2.2 如何做趋势-季节分解

Autoformer 使用移动平均提取趋势：

$$
X_t=\operatorname{AvgPool}(\operatorname{Padding}(X))
$$

$$
X_s=X-X_t
$$

其中：

- $X_t$ 是 trend-cyclical component，即趋势-周期成分；
- $X_s$ 是 seasonal component，即去除趋势后的残差成分。

这个分解更准确地说是“平滑项 + 残差项”，不是严格的频谱高低频分解。移动平均具有低通效果，因此 $X_s$ 通常包含更快的变化，但也可能包含噪声、趋势提取误差和其他未建模成分。

### 2.3 不是一次分解，而是渐进分解

Autoformer 的关键不是公式本身，而是把 `SeriesDecomp` 插入每个编码器和解码器层。

编码器中，每次经过 Auto-Correlation 或前馈网络后都执行一次分解：

$$
S_{en}^{l,1},T_{en}^{l,1}=
\operatorname{SeriesDecomp}
\left(
\operatorname{AutoCorrelation}(X_{en}^{l-1})+X_{en}^{l-1}
\right)
$$

$$
S_{en}^{l,2},T_{en}^{l,2}=
\operatorname{SeriesDecomp}
\left(
\operatorname{FeedForward}(S_{en}^{l,1})+S_{en}^{l,1}
\right)
$$

编码器只把季节/残差部分继续向后传递，被提取的趋势项不再进入编码器主干。因此编码器逐层排除平滑趋势，集中建模周期性和变化部分。

解码器同时维护：

- 季节主干 $S_{de}$；
- 趋势累积量 $T_{de}$。

每个解码层会多次执行分解，并把每次提取出的趋势分量投影后累加：

$$
T_{de}^{l}=T_{de}^{l-1}
+W_{l,1}T_{de}^{l,1}
+W_{l,2}T_{de}^{l,2}
+W_{l,3}T_{de}^{l,3}
$$

最终预测是季节预测和累计趋势预测之和：

$$
\hat{Y}=W_sX_{de}^{M}+T_{de}^{M}
$$

### 2.4 解码输入如何初始化

Autoformer 对输入后半段进行一次分解，并使用不同的未来占位符：

$$
X_{de}^{s}=\operatorname{Concat}(X_{en}^{s},0)
$$

$$
X_{de}^{t}=\operatorname{Concat}(X_{en}^{t},\operatorname{Mean}(X_{en}))
$$

季节部分的未来使用零占位，趋势部分的未来使用历史均值占位。这样趋势预测从平稳基线开始，季节预测则从无偏的零残差开始。

### 2.5 Auto-Correlation 如何建模季节动态

Auto-Correlation 不做普通 token-to-token 注意力，而是：

1. 使用 FFT 计算序列与其不同时间延迟版本之间的相关性；
2. 选择相关性最大的 top-$k$ 个延迟；
3. 按这些延迟对 Value 序列进行循环移位；
4. 根据时间延迟相关性加权聚合。

因此它捕捉的是“相同周期位置的相似子序列”，比逐点注意力更符合长期周期序列的结构。

### 2.6 对 DeCoF 的启发

Autoformer 最接近 DeCoF 中的残差路线：

$$
H_r=H-H_c
$$

它说明：只要 $H_c$ 是可信的平滑粗成分，残差就可以单独作为细粒度建模对象，不需要额外的偏差原型才能形成完整模型。

但 Autoformer 与 DeCoF 有一个重要区别：Autoformer 的趋势项由明确的移动平均产生，而 DeCoF 当前的 $H_c$ 由低秩注意力产生。低秩约束只能说明表示被压缩，并不能直接说明它是低频趋势。

## 3. FEDformer

论文：*FEDformer: Frequency Enhanced Decomposed Transformer for Long-term Series Forecasting*。

### 3.1 与 Autoformer 的关系

FEDformer 延续了 Autoformer 的深度趋势-季节分解框架，但进行了两项升级：

1. 用多尺度 Mixture-of-Experts 分解替代单一窗口移动平均；
2. 在 Fourier 或 Wavelet 空间中进行表示学习，用 Frequency Enhanced Block 和 Frequency Enhanced Attention 替代时间域注意力。

所以 FEDformer 同时包含两种不同意义上的“分解”：

- **趋势-季节分解**：决定主干中哪些信息属于趋势，哪些属于季节残差；
- **频域表示分解**：决定如何高效表示和交互序列模式。

不能把这两部分混为一个操作。

### 3.2 MOE Decomposition

固定窗口移动平均可能无法覆盖多种周期。FEDformer 使用多个不同窗口的平均池化器：

$$
X_{trend}=
\operatorname{Softmax}(L(X))
\ast F(X)
$$

其中：

- $F(X)$ 表示多个不同窗口平均滤波器的输出；
- $L(X)$ 根据当前输入生成数据相关的专家权重；
- 不同趋势尺度被自适应组合为最终趋势。

季节项仍然通过残差获得：

$$
X_{seasonal}=X-X_{trend}
$$

这比单一移动平均更适合包含多个周期尺度的序列。

### 3.3 Fourier 版本

Frequency Enhanced Block 的 Fourier 版本首先将时间序列映射到频域：

$$
Q=\mathcal{F}(q)
$$

随后随机保留 $M$ 个 Fourier modes：

$$
\tilde{Q}=\operatorname{Select}(\mathcal{F}(q)),
\qquad M\ll L
$$

保留的 modes 并不只是最低频率，而是可能同时包含低频和高频。论文明确指出：

- 只保留低频可能丢失重要事件；
- 保留全部高频可能过拟合噪声；
- 选择少量频率成分可以在信息保留和复杂度之间折中。

选中的频率成分与可学习复数核进行交互，补零后再逆变换回时间域：

$$
\operatorname{FEB_f}(q)=
\mathcal{F}^{-1}
\left(
\operatorname{Padding}(\tilde{Q}\odot R)
\right)
$$

Frequency Enhanced Attention 则在频域中计算 Query、Key 和 Value 的交互：

$$
\operatorname{FEA_f}(q,k,v)=
\mathcal{F}^{-1}
\left(
\operatorname{Padding}
\left(
\sigma(\tilde{Q}\tilde{K}^{\top})\tilde{V}
\right)
\right)
$$

在固定 mode 数量下，其复杂度可近似为 $O(L)$。

### 3.4 Wavelet 版本

FEDformer-w 使用多小波表示。相较 Fourier，Wavelet 同时具有时间和频率局部性，因此更适合捕获局部突变。

其基本过程是：

1. 递归执行多尺度小波分解；
2. 每层得到高频部分、低频部分和需要继续分解的粗尺度部分；
3. 分别使用频率增强模块处理这些部分；
4. 通过逆向递归逐层重构回原时间长度。

需要注意，FEDformer-w 的小波分解发生在 Transformer 表示模块内部。它最终仍然服务于趋势-季节深度架构，并没有像 STWave 那样把低频和高频定义为两个长期独立的业务分支。

### 3.5 最终融合方式

FEDformer 与 Autoformer 类似：季节项在主干中深度变换，趋势项逐层累积，最终输出为两者相加：

$$
\hat{Y}=W_sX_{de}^{M}+T_{de}^{M}
$$

它没有为趋势项和季节项分别设计完全不同的预测器，也没有使用偏差感知门控。

### 3.6 对 DeCoF 的启发

FEDformer 对 DeCoF 有两点直接启发：

1. 粗粒度表示最好允许多尺度。单一低秩投影可能只覆盖一种粗尺度，可以考虑多秩或多窗口专家。
2. “选少量频率成分”与“低秩投影”都能形成紧凑表示，但两者不能等价。若论文将 $H_c$ 称为低频成分，应补充频谱或平滑性证据。

## 4. STWave

论文：*When Spatio-Temporal Meet Wavelets: Disentangled Traffic Forecasting via Efficient Spectral Graph Attention Networks*。

STWave 是四篇中与 DeCoF 粗细分支最接近的模型，因为它显式构造低频趋势通道和高频事件通道，并为二者使用不同的时空编码器。

### 4.1 使用多层 DWT 显式分离趋势和事件

给定交通序列 $X$，STWave 使用多层离散小波变换。以两层 DWT 为例：

$$
\bar{X}_{2,l}
=
\left(g\star(g\star X)_{\downarrow 2}\right)_{\downarrow 2}
$$

$$
\bar{X}_{2,h}
=
\left(h\star(g\star X)_{\downarrow 2}\right)_{\downarrow 2}
$$

$$
\bar{X}_{1,h}
=
(h\star X)_{\downarrow 2}
$$

其中：

- $g$ 是低通滤波器；
- $h$ 是高通滤波器；
- $\downarrow 2$ 是二倍下采样；
- $\bar{X}_{2,l}$ 是低频趋势；
- $\bar{X}_{2,h}$ 和 $\bar{X}_{1,h}$ 是不同尺度的高频事件。

为了恢复到原时间长度，STWave 使用 IDWT 上采样和逆滤波。

最终趋势表示为：

$$
X_l=
W_g
\left[
g^T\star
\left(g^T\star\bar{X}_{2,l}\right)_{\uparrow 2}
\right]_{\uparrow 2}
+b_g
$$

最终事件表示把所有尺度的高频重构相加：

$$
X_h=
W_h
\left[
g^T\star
\left(h^T\star\bar{X}_{2,h}\right)_{\uparrow 2}
+
h^T\star\bar{X}_{1,h}
\right]
+b_h
$$

STWave 没有为每个高频尺度建立独立预测分支，而是把多层高频统一重构并加总成一个 Event 通道。这是在信息保留和计算成本之间的折中。

### 4.2 趋势和事件使用不同的时间模型

STWave 根据两类信号的性质选择不同模块：

| 成分 | 时间性质 | 时间建模模块 |
| --- | --- | --- |
| Trend $X_l$ | 平稳、长期相关 | Temporal Attention |
| Event $X_h$ | 突发、局部连续 | 小卷积核 Causal Convolution |

事件分支使用因果卷积：

$$
X_h^{conv}=\operatorname{ReLU}(\theta\star X_h)
$$

趋势分支使用全局时间注意力，使每个时间位置可以观察较长历史：

$$
X_l^{tatt}=\operatorname{TemporalAttention}(X_l)
$$

这个设计比“两个分支使用完全相同的网络”更有明确的归纳偏置：长期稳定趋势需要全局感受野，突发事件更需要局部敏感性。

### 4.3 两个通道分别建模空间关系

趋势和事件经过各自时间模块后，分别进入 Efficient Spectral Graph Attention Network。也就是说，STWave 并不假设趋势空间关系与事件空间关系相同。

其空间模块同时使用：

- 全图注意力建模动态远距离关系；
- 图小波位置编码注入图结构和局部性；
- Query Sampling 将空间注意力复杂度降低到约 $O(N\log N)$。

### 4.4 不是简单把趋势与事件相加

STWave 首先分别预测未来趋势和未来事件：

$$
\hat{Y}_l^f=\operatorname{Predictor}_l(Z_l)
$$

$$
\hat{Y}_h^f=\operatorname{Predictor}_h(Z_h)
$$

随后使用 Adaptive Event Fusion。其思想是：

- 趋势预测通常较稳定；
- 事件预测容易受到分布漂移影响；
- 应保留可信事件并抑制错误事件，而不是无条件相加。

融合模块以趋势状态作为稳定基准，通过因果注意力从预测事件中选择有用修正，最终形成：

$$
\hat{Y}=hat{Y}_l^f+\operatorname{AdaptiveEvent}(\hat{Y}_l^f,\hat{Y}_h^f)
$$

论文消融中的 `w/o AF` 将自适应融合替换成直接加法，在所有报告数据集上都比完整 STWave 差。这对 DeCoF 的 $H_f=H_r+H_d$ 很有参考意义：如果偏差或事件具有较大不确定性，直接相加可能放大错误动态。

### 4.5 多重监督

STWave 同时优化：

- 最终交通预测损失 $\mathcal{L}_{flow}$；
- 趋势辅助损失 $\mathcal{L}_{trend}$。

$$
\mathcal{L}=\mathcal{L}_{flow}+\mathcal{L}_{trend}
$$

趋势辅助监督能够稳定低频分支，降低事件分布漂移对整体预测的干扰。论文并没有单独要求事件预测完全重构真实高频项，而是让事件分支通过最终预测损失学习有用修正。

### 4.6 对 DeCoF 的启发

STWave 给 DeCoF 的启发最直接：

1. 高频动态可以只来自一个明确的高频来源，不需要同时使用隐空间残差和历史偏差原型。
2. 高频分支适合使用局部模块，粗分支适合使用全局模块。
3. 粗细融合最好由稳定粗成分引导，对高频修正进行筛选。
4. 可以给粗分支增加辅助监督，防止粗成分被细分支污染。

## 5. STDN

论文：*Spatiotemporal-aware Trend-Seasonality Decomposition Network for Traffic Flow Forecasting*。

STDN 不使用固定移动平均或 DWT，而是根据时间位置和空间位置学习每个节点、每个时刻的趋势比例。

### 5.1 先学习动态关系图

STDN 使用时间槽嵌入、起始节点嵌入、终止节点嵌入和核心张量构造时变邻接关系：

$$
A'_{t,i,j}
=
\sum_{o,q,r}
K_{o,q,r}E^t_{t,o}E^e_{i,q}E^s_{j,r}
$$

然后经过 ReLU 和归一化得到每个时间片的动态邻接矩阵 $A_t$。交通输入经过动态图卷积：

$$
H_L=
\sum_{l=0}^{L}
(A_t)^lH_l^tW_l
$$

得到经过动态空间关系建模的隐表示：

$$
H_L\in\mathbb{R}^{T\times N\times D}
$$

### 5.2 学习时空上下文嵌入

时间嵌入来自：

- Time of Day；
- Day of Week。

时间特征经过 one-hot、MLP、ReLU 和 Sigmoid 得到 $M_h^t$。空间嵌入则来自道路图归一化 Laplacian 的若干最小非平凡特征向量，再经过 MLP 得到 $M^s$。

广播到相同形状后组合为：

$$
M=\sin(M_h^t)+\operatorname{ReLU}(M^s)
$$

因此 $M$ 表示每个时间、每个节点、每个隐通道的时空上下文。

### 5.3 使用乘性门控提取趋势

STDN 的趋势-季节分解是：

$$
X_t=H_L\odot M
$$

$$
X_s=H_L-X_t
$$

其中：

- $X_t$ 是趋势成分；
- $X_s$ 是季节/残差成分。

它与 Autoformer 的差异是：

- Autoformer 的趋势来自时间移动平均；
- STDN 的趋势来自时空上下文门控；
- 同一个节点在不同时间可以使用不同的趋势比例；
- 不同节点可以具有不同的趋势定义。

这使 STDN 更适合空间异质的交通网络。

但它不是严格的频率分解。由于 $M$ 也不一定限制在 $[0,1]$，$X_t$ 与 $X_s$ 不一定是凸分解，也不保证正交。它更准确地说是“时空条件化的隐表示分解”。

### 5.4 两部分分别编码再相加

趋势和季节项分别使用 GRU：

$$
Y_t=\operatorname{GRU}_t(X_t)
$$

$$
Y_s=\operatorname{GRU}_s(X_s)
$$

随后逐元素相加：

$$
Y=Y_t+Y_s
$$

再送入 Bottleneck Transformer Decoder。解码器还拼接未来时间嵌入和空间嵌入，让预测显式感知未来时刻和节点位置。

值得注意的是，STDN 虽然使用两个 GRU 编码器，但两条分支采用相同类型的网络结构，没有像 STWave 那样为趋势和事件使用不同的时间建模机制。

### 5.5 对 DeCoF 的启发

STDN 与 DeCoF 的残差分支最相似：

$$
X_s=H_L-H_L\odot M
$$

对应 DeCoF：

$$
H_r=H-H_c
$$

STDN 没有额外的 $H_d$，说明单独使用隐表示残差作为细粒度动态在结构上完全可行。

同时，STDN 提示 DeCoF 可以让粗粒度提取依赖时空上下文，而不仅由一个全局低秩注意力产生。例如：

$$
H_c=\operatorname{LRPA}(H)\odot M_{st}
$$

或让 $M_{st}$ 控制粗细融合门控。

## 6. 四篇论文的数据流对比

### 6.1 Autoformer

```text
Input
  -> Moving Average Decomposition
  -> Seasonal Residual Main Path
  -> Auto-Correlation + Progressive Decomposition
  -> Seasonal Prediction

Extracted Trend at Every Decoder Layer
  -> Progressive Accumulation
  -> Trend Prediction

Seasonal Prediction + Trend Prediction
  -> Final Forecast
```

### 6.2 FEDformer

```text
Input
  -> Multi-scale MOE Decomposition
  -> Seasonal Residual Main Path
  -> Fourier/Wavelet Enhanced Blocks
  -> Seasonal Prediction

Multi-scale Trend Components
  -> Progressive Accumulation
  -> Trend Prediction

Seasonal Prediction + Trend Prediction
  -> Final Forecast
```

### 6.3 STWave

```text
Input
  -> Multi-level DWT
  -> Low-frequency Reconstruction -> Trend
  -> Multi-high-frequency Reconstruction and Sum -> Event

Trend -> Temporal Attention -> ESGAT -> Trend Predictor
Event -> Causal Convolution -> ESGAT -> Event Predictor

Trend Prediction + Adaptively Selected Event Correction
  -> Final Forecast
```

### 6.4 STDN

```text
Input
  -> Dynamic Graph Convolution -> H_L

Time Embedding + Spatial Embedding -> M

H_L * M       -> Trend       -> GRU_t
H_L - H_L * M -> Seasonality -> GRU_s

GRU_t + GRU_s
  -> Bottleneck Transformer Decoder
  -> Final Forecast
```

## 7. 与当前 DeCoF 高频分支的对应关系

DeCoF 当前定义为：

$$
H_c=\operatorname{LRPA}(H)
$$

$$
H_r=H-\operatorname{stopgrad}(H_c)
$$

$$
D=X-\operatorname{Anchor}(X)
$$

$$
H_d=\operatorname{PrototypeEncoder}(D)
$$

$$
H_f=H_r+H_d
$$

对应关系如下：

| DeCoF 组件 | 最接近的已有设计 | 主要区别 |
| --- | --- | --- |
| $H_c$ | Autoformer/FEDformer 趋势；STDN 的 $X_t$ | DeCoF 使用低秩注意力，没有显式低通或趋势门控 |
| $H_r=H-H_c$ | Autoformer/FEDformer 的 seasonal residual；STDN 的 $X_s$ | DeCoF 使用 `stopgrad`，三篇论文没有相同设置 |
| 历史锚点 $D=X-A$ | 四篇中没有完全对应设计 | 属于 DeCoF 的额外显式偏差来源 |
| 原型编码 $H_d$ | 四篇中没有完全对应设计 | 属于 DeCoF 的潜在创新点 |
| 高频专用 Fold | STWave 的 causal convolution + ESGAT | STWave 明确按信号性质设计不同网络 |
| 偏差感知融合 | STWave Adaptive Event Fusion | STWave 用趋势筛选事件，而不是无条件相加 |

### 7.1 $H_r$ 和 $H_d$ 是否必须同时存在

从四篇论文看，答案是否定的。

- Autoformer：只使用趋势残差作为季节项；
- FEDformer：只使用 MOE 趋势残差作为季节项；
- STWave：只使用 DWT 高频重构作为事件项；
- STDN：只使用时空趋势门控后的残差作为季节项。

没有一篇论文同时把“隐空间粗粒度残差”和“历史锚点偏差编码”相加后才定义高频分支。因此，同时使用 $H_r$ 与 $H_d$ 是 DeCoF 自己需要验证的研究假设，而不是已有分解模型的必要条件。

### 7.2 直接相加是否有充分依据

直接相加：

$$
H_f=H_r+H_d
$$

只有在以下条件成立时比较合理：

1. $H_r$ 与 $H_d$ 处于相同隐空间；
2. 两者数值尺度接近；
3. 两者没有严重重复描述相同变化；
4. 历史锚点偏差可靠；
5. 偏差原型产生错误信息时，模型能够抑制其影响。

STWave 的消融结果说明，对不稳定事件无条件相加并不是最优选择。因此更推荐：

$$
\tilde{H}_r=\operatorname{LN}(H_r)
$$

$$
\tilde{H}_d=\operatorname{LN}(W_dH_d)
$$

$$
G_d=\sigma
\left(
\operatorname{MLP}
([\tilde{H}_r;\tilde{H}_d;s_d])
\right)
$$

$$
H_f=H_r+G_d\odot W_dH_d
$$

或者使用 Concat 学习修正量：

$$
H_f=H_r+
\operatorname{MLP}
([\operatorname{LN}(H_r);\operatorname{LN}(H_d);s_d])
$$

### 7.3 当前最大理论风险

当前 DeCoF 最大的问题不是 $H_r$ 与 $H_d$ 如何融合，而是：

> 低秩注意力输出 $H_c$ 是否真的可以解释为低频或粗粒度成分。

四篇论文都给出了明确的粗成分产生机制：

- Autoformer：移动平均；
- FEDformer：多窗口平均专家；
- STWave：DWT 低频滤波；
- STDN：时空条件乘性门控。

而低秩注意力只保证有限秩或紧凑表示，可能同时保留低频趋势、显著高频事件和空间异常。如果 $H_c$ 不是 $H$ 的同空间粗粒度重构，那么：

$$
H-H_c
$$

也不一定具有清晰的高频残差意义。

建议至少加入以下一项证据或约束：

- 比较 $H_c$ 与 $H_r$ 的频谱能量分布；
- 对 $H_c$ 加时间平滑正则；
- 对 $H_c$ 增加粗粒度重构损失；
- 让 $H_c$ 逼近移动平均或小波低频教师信号；
- 将“高频分支”暂时命名为“细粒度残差分支”，避免过强频率表述。

## 8. 推荐的 DeCoF 实验路线

### 8.1 第一阶段：确定高频来源是否必要

比较三个版本：

$$
\text{Residual-only:}\quad H_f=H_r
$$

$$
\text{Deviation-only:}\quad H_f=H_d
$$

$$
\text{Combined:}\quad H_f=H_r+G_d\odot H_d
$$

这一步回答 $H_r$ 与 $H_d$ 是否互补。

### 8.2 第二阶段：比较融合方式

在确认二者互补后，比较：

1. Add：$H_r+H_d$；
2. Concat + Linear：$W[H_r;H_d]$；
3. Residual MLP：$H_r+\operatorname{MLP}([H_r;H_d])$；
4. Gated Injection：$H_r+G_d\odot H_d$。

### 8.3 第三阶段：验证粗细分工

需要报告：

- $H_c$、$H_r$、$H_d$ 的频谱能量；
- 三者的平均范数；
- $H_r$ 与 $H_d$ 的相关性或 CKA；
- 门控值与偏差分数的关系；
- 低偏差、中偏差和高偏差样本上的误差；
- 是否存在某个分支被长期忽略。

### 8.4 第四阶段：借鉴 STWave 的分支专门化

建议粗细 `Fold` 不完全相同：

- Coarse Fold：全局时间注意力、大窗口、较强下采样；
- Fine Fold：局部因果卷积、小窗口、较弱下采样；
- 两条分支分别学习空间关系，再在更高层融合。

这比只改变输入而使用完全相同的编码器，更容易证明粗细分支具有不同作用。

## 9. 最终建议

结合四篇论文，当前 DeCoF 最稳妥的最小版本是：

$$
H_c=\operatorname{CoarseEncoder}(H)
$$

$$
H_f=H-\operatorname{stopgrad}(H_c)
$$

$$
Z_c=\operatorname{GlobalFold}(H_c)
$$

$$
Z_f=\operatorname{LocalFold}(H_f)
$$

$$
Z=Z_c+G_f\odot Z_f
$$

在这个可解释的基础版本上，再加入 Historical Anchor 和 Deviation Prototype Encoder：

$$
H_f=H_r+G_d\odot H_d
$$

这样能把 $H_d$ 的贡献作为清晰的增量创新进行验证。

如果一开始就同时引入 $H_r$、$H_d$、偏差分数、可见性和门控融合，当性能提高时很难判断增益究竟来自哪一项，也很难证明 $H_r$ 与 $H_d$ 确实互补。

从论文定位角度，可以将核心叙事表述为：

> 现有方法主要使用时间平滑、频域滤波或时空门控得到单一残差动态。DeCoF 在隐空间粗细分解之外，引入相对于历史锚点的偏差原型，将“模型未解释残差”和“历史常态偏离”区分开，并通过偏差感知门控选择性融合二者。

但要成立，需要通过 Residual-only、Deviation-only 和 Combined 三组实验明确证明两类动态信息不是重复的。

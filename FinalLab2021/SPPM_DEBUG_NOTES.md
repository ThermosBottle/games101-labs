# SPPM 开发与调试笔记

本文记录在 CUDA 路径追踪器中加入 SPPM（Stochastic Progressive Photon Mapping）后的开发和调试过程。

## 1. 实现目标

- 保留原有的路径追踪和 MIS 渲染模式。
- 为 SPPM 增加相机子路径、光子子路径、光子空间索引和密度估计流程。
- 支持漫反射、镜面反射和理想玻璃等不同事件类型。
- 在低样本数下优先保证状态一致性、数值有效性和输出可验证性。

## 2. SPPM 数据流

每次迭代执行以下步骤：

1. 相机 kernel 为每个像素生成一条相机路径。
2. 相机路径遇到第一个可用于密度估计的漫反射顶点时，写入或更新 `SPPMPixel`。
3. 光子 kernel 从光源发射光子，并沿路径传播。
4. 光子在可用于密度估计的表面顶点记录位置、方向和功率。
5. 主机端根据光子位置建立 KD-tree，并上传到 GPU。
6. gather kernel 在每个可见点的半径内查询光子。
7. update kernel 更新光子计数、通量和半径。
8. resolve kernel 将累计量转换为图像像素。

设当前可见点的搜索半径为 $R$，累计光子数为 $N$，累计通量为 $\tau$，总发射光子数为 $M$，则解析阶段的基本形式为

\[
L \approx L_{\mathrm{emitted}}
  + \frac{\tau}{\pi R^2 M}.
\]

`tau` 不是最终辐射亮度，不能在 gather 阶段直接写入 framebuffer。

## 3. 主要状态及不变量

### 3.1 `SPPMPixel`

- `position`：当前可见点位置。
- `normal`：用于半球判断的方向，必须与 BSDF 使用的方向约定一致。
- `viewDirection`：相机射线方向；若 BSDF 的出射方向定义为“离开表面”，通常需要使用 `-viewDirection`。
- `beta`：相机子路径到可见点的 throughput。
- `tau`：跨迭代累计的局部反射通量。
- `radiusSquared`：当前搜索半径的平方。
- `photonCount`：累计有效光子数。
- `newFlux` 和 `newPhotonCount`：当前迭代的临时量，每轮相机阶段开始时清零。
- `valid`：表示当前是否存在可用于 gather 的可见点。

需要保持以下不变量：

\[
R^2 > 0, \qquad N \geq 0, \qquad M \geq 0.
\]

所有参与 resolve 的向量和标量都应为有限值：

\[
\operatorname{isfinite}(x) = \mathrm{true}.
\]

### 3.2 Photon

光子记录至少包含：

- 位置 $\mathbf{x}$；
- 入射方向 $\omega_i$；
- 已包含光子路径 throughput 的功率 $\Phi$；
- 有效标志 `valid`。

如果 `photon.power` 已经包含光子路径上的入射余弦项，gather 时不应再次乘入同一个入射余弦项。

## 4. 开发顺序

1. 先实现单次迭代的光子发射和记录。
2. 用线性遍历替代 KD-tree，验证光子位置、方向和功率。
3. 加入 KD-tree，并将 KD-tree 查询结果与线性遍历比较。
4. 加入单像素或小图像的 gather。
5. 加入 `tau`、半径和光子数的 progressive update。
6. 加入 resolve 和 PPM 输出。
7. 最后接入玻璃、镜面和俄罗斯轮盘赌。
8. 每加入一种事件类型，都与 MIS 模式进行独立回归。

## 5. 调试重点

### 5.1 方向约定

代码中相机射线方向是从相机指向表面。若 BSDF 使用的方向均为离开表面的方向，则：

\[
\omega_o = -\,\omega_{\mathrm{camera}}.
\]

光子方向若记录为指向表面的入射方向，则 gather 的可见半球条件为

\[
\mathbf{n}\cdot(-\omega_{\mathrm{photon}}) > 0.
\]

同时还应检查相机出射方向：

\[
\mathbf{n}\cdot\omega_o > 0.
\]

只检查光子方向而不检查相机方向，会允许不完整的双向半球组合进入估计器。

### 5.2 法线和射线偏移

表面法线先根据当前射线方向翻转为朝向入射半空间：

\[
\mathbf{n}\cdot\omega_{\mathrm{ray}} \leq 0.
\]

生成下一条射线时，根据新方向选择偏移方向：

\[
\mathbf{o}_{\mathrm{new}}
 = \mathbf{x} + \operatorname{sign}(\omega_{\mathrm{new}}\cdot\mathbf{n})\,\epsilon\mathbf{n}.
\]

偏移不一致会产生自相交、错误遮挡和边缘黑点。调试时应分别记录相机射线、光子射线和 shadow ray 的起点。

### 5.3 Delta 事件和折射

镜面和理想玻璃不应作为普通漫反射顶点进行 photon gather。它们只用于路径延续。

玻璃折射的关键量是入射与出射介质折射率比：

\[
\eta = \frac{\eta_i}{\eta_t}.
\]

对于透射路径，throughput 权重必须与当前路径测量约定一致。实现中应单独检查：

- 进入和离开玻璃时的 $\eta_i,\eta_t$ 是否交换；
- 全反射时是否只保留反射分支；
- Fresnel 分支概率是否与权重除法匹配；
- 折射方向是否位于正确的表面半空间；
- 透射权重是否产生非有限值。

### 5.4 Persistent visible point

SPPM 的可见点状态跨迭代存在。某次相机路径失败时，不应自动清除之前已经有效的状态，否则可能发生：

\[
\text{有效历史状态}
\rightarrow \text{本轮偶发 miss}
\rightarrow \text{黑色 resolve 结果}.
\]

当前实现采用以下策略：

- 每轮清零 `newFlux` 和 `newPhotonCount`；
- 当前路径成功到达漫反射点时更新 visible point；
- 当前路径失败时保留已有的 persistent 状态。

这不会修复错误的可见点，但可以避免一次偶发失败破坏已有累计结果。

## 6. 典型问题与定位

### 6.1 玻璃球出现彩色像素噪声

首先检查相机路径的随机种子。如果每个像素在所有迭代使用相同种子，玻璃 Fresnel 的一次反射/折射分支会被永久固定，导致结构化彩色噪声。

SPPM 相机路径应允许跨迭代变化，例如使用像素索引和迭代索引共同生成种子：

\[
s = H(p \oplus k i),
\]

其中 $p$ 是像素索引，$i$ 是迭代索引，$H$ 是 hash 函数。

### 6.2 漫反射物体边缘黑点

按以下顺序排查：

1. visible point 是否在本轮失败时被错误置为无效；
2. 相机出射方向和光子入射方向是否都通过半球判断；
3. 光子和相机射线的 offset 是否一致；
4. gather 的 BSDF 是否返回负值或非有限值；
5. `tau`、半径和 photon count 是否被错误更新；
6. KD-tree 查询是否漏查半径边界内的节点。


## 7. 当前实现中的结论

- 相机路径的固定 per-pixel seed 会冻结玻璃 Fresnel 分支，不适合作为跨迭代的 SPPM 相机采样策略。
- 相机路径失败不应覆盖之前有效的 persistent visible point。
- 玻璃球反射左侧红墙的颜色在物理上是合理现象。
- SPPM 比 MIS 更早显示某些困难的间接颜色，不代表 MIS 必然错误；MIS 需要更多样本才能稳定采样 delta 反射后的漫反射路径。
- 低迭代数下的黑像素比例不能单独作为正确性结论，还需要结合路径命中率、可见点有效率和有限值检查。
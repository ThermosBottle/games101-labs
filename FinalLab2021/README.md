# CUDA 光线追踪与 SPPM

本项目实现了一个面向 Cornell Box 场景的 GPU 光线渲染器，并提供两种互补的全局光照估计方法：基于多重重要性采样（Multiple Importance Sampling, MIS）的路径追踪，以及随机渐进式光子映射（Stochastic Progressive Photon Mapping, SPPM）。项目同时保留 CPU 光线追踪器和 CUDA 几何验证程序，用于结果对照与实现验证。

## 1. 项目目标与特性

- 使用 CUDA 实现三角形、球体和线性 BVH 的光线求交；
- 支持漫反射、GGX 微表面、理想镜面和理想介电玻璃；
- 支持面积光源采样、直接光照、路径延续和俄罗斯轮盘赌；
- 以 P6 PPM 格式输出图像；
- 通过 MIS 和 SPPM 分别展示路径空间采样与光子密度估计的特点；
- 提供 CUDA 求交验证、场景上传和光子 KD-tree 等辅助模块。

当前 CUDA 示例场景由 Cornell Box、面积光源、玻璃球和镜面墙组成。场景几何来自 `models/cornellbox/`；程序应从项目根目录或能够正确解析该相对路径的位置运行。

## 2. 数学原理

### 2.1 渲染方程与路径积分

在表面点 $x$，出射辐亮度满足渲染方程

$$
L_o(x,\omega_o)=L_e(x,\omega_o)+
\int_{\mathcal{H}^2} f_r(x,\omega_i,\omega_o)
L_i(x,\omega_i)\,|\cos\theta_i|\,\mathrm{d}\omega_i.
$$

程序以随机路径近似该积分。设路径采样概率为 $p(\omega_i)$，则一次非 delta 路径延续使用估计量

$$
\beta' = \beta\, f_r(\omega_i,\omega_o)
\frac{|\cos\theta_o|}{p(\omega_o)\,p_{\mathrm{survival}}},
$$

其中 $\beta$ 是当前路径吞吐量，$p_{\mathrm{survival}}$ 是俄罗斯轮盘赌的存活概率。程序在路径深度达到 3 后启用轮盘赌，并对存活路径除以相应概率，以保持估计的无偏性。

### 2.2 材质模型

漫反射材质采用 Lambert 模型

$$
f_d=\frac{K_d}{\pi}.
$$

其方向采用余弦加权半球采样，概率密度为

$$
p_d(\omega_o)=\frac{\max(0,\mathbf n\cdot\omega_o)}{\pi}.
$$

微表面材质采用 Cook--Torrance GGX 模型

$$
f_r=\frac{D(\mathbf h)G(\omega_i,\omega_o)F(\omega_i,\mathbf h)}
{4(\mathbf n\cdot\omega_i)(\mathbf n\cdot\omega_o)},
$$

其中 $D$ 为 GGX 法线分布，$G$ 为遮蔽--阴影项，$F$ 为 Schlick Fresnel 项。程序以基于高光颜色亮度的概率混合漫反射采样和微表面采样，并使用与半向量变换一致的 PDF：

$$
p_{\mathrm{GGX}}(\omega_o)=
\frac{D(\mathbf h)(\mathbf n\cdot\mathbf h)}{4|\omega_i\cdot\mathbf h|}.
$$

镜面与玻璃属于 delta 事件，不使用普通立体角 PDF，而是直接生成唯一的反射或折射方向。玻璃的 Fresnel 反射率由介质折射率 $\eta_i,\eta_t$ 和入射余弦计算；透射分支使用

$$
\eta=\frac{\eta_i}{\eta_t},\qquad
w_t=\frac{\eta^2}{1-F_r},
$$

并在全反射时只保留反射分支。

### 2.3 MIS 路径追踪

MIS 模式在每个非发光表面点采样面积光源，并将面积 PDF 转换为立体角 PDF：

$$
p_L(\omega)=p_A\frac{\|x_l-x\|^2}{|\mathbf n_l\cdot(-\omega)|}.
$$

代码采用 balance heuristic 的平方形式计算权重

$$
w_L=\frac{p_L^2}{p_L^2+p_B^2},
$$

其中 $p_B$ 为 BSDF PDF。阴影射线负责可见性测试，直接光照贡献为

$$
L_{\mathrm{direct}}=\beta f_r L_e\frac{|\mathbf n\cdot\omega|w_L}{p_L}.
$$

### 2.4 SPPM 密度估计

SPPM 将一次迭代分为相机子路径和光子子路径。相机路径到达第一个具有漫反射 lobes 的顶点后记录可见点；镜面和玻璃顶点仅用于路径延续。光子路径在每个漫反射顶点记录位置、入射方向和已经包含路径吞吐量的功率 $\Phi$。

设可见点搜索半径为 $R$，累计局部通量为 $\tau$，累计有效光子数为 $N$，迄今发射光子总数为 $M$，则

$$
L_{\mathrm{SPPM}}\approx L_{\mathrm{camera\ emission}}
 +\frac{\tau}{\pi R^2M}.
$$

当前渐进更新为

$$
\gamma=\frac{N+\alpha M_k}{N+M_k},\qquad
R^2\leftarrow\gamma R^2,\qquad
\tau\leftarrow\gamma(\tau+\Delta\tau),
$$

其中 $M_k$ 和 $\Delta\tau$ 是当前迭代的新命中光子数和新通量，$\alpha=0.7$。无新光子时令 $\gamma=1$，以保留初始搜索半径。

光子按发射面积 PDF $p_A$ 和余弦半球方向采样。由于余弦 PDF 为 $\cos\theta/\pi$，初始光子功率为

$$
\Phi_0=L_e\frac{\pi}{p_A}.
$$

gather 阶段只在光子位于搜索圆盘内且位于可见点入射半球时累加

$$
\Delta\tau\mathrel{+}=\beta\,\Phi\,f_r(\omega_i,\omega_o).
$$

`tau` 是累计局部通量而非最终颜色，归一化只能在 resolve 阶段进行。

## 3. 程序架构

### 3.1 数据组织

`Scene`、`Material`、几何对象和 BVH 在主机端组织场景。`CudaScene` 将其展平为无指针数组并上传 `CudaTriangle`、`CudaSphere`、`CudaPrimitive`、`CudaBvhNode` 和 `CudaMaterial`。`CudaSceneView` 以非拥有视图封装设备指针和数量，按值传入 kernel。`CudaTypes.cuh` 中的共享结构仅含可平凡拷贝字段，避免 GPU 端进行指针修复或虚函数分派。

### 3.2 核心模块

| 模块 | 职责 |
| --- | --- |
| `CudaKernels.cu/.cuh` | 求交、BSDF、路径追踪及 SPPM kernels |
| `CudaScene.cpp/.cu` | 场景展平、BVH 与材质上传 |
| `CudaPhotonMap.cpp/.cuh` | 主机端构造光子 KD-tree 并上传 |
| `CudaPathTracing.cu` | 参数解析、场景创建、迭代调度和 PPM 写出 |
| `CudaValidation.cu` | 线性求交与 BVH 求交验证 |
| `Scene.cpp`、`Material.hpp` | CPU 场景和材质参考实现 |

正式渲染使用 BVH 求交；验证程序将其与线性遍历的命中、距离和法线结果比较。光子 KD-tree 在每次 SPPM 迭代由主机端重建，再供 GPU gather 查询。

## 4. 运行逻辑

程序入口为 `CudaPathTracing`，参数形式为：

```text
CudaPathTracing [iterations] [width] [height] [maxDepth]
                [diffuse|microfacet] [mis|sppm] [output.ppm]
```

默认值为 `16 784 784 8 diffuse mis`。程序先构造 Cornell Box、面积光源、玻璃球和镜面墙，建立 BVH 并上传设备数据；随后按迭代执行以下分支：

- **MIS**：一次 GPU 路径追踪，累加每像素辐亮度，写出前除以迭代数；
- **SPPM**：相机 kernel → 光子发射 → 主机 KD-tree 构建/上传 → GPU gather → progressive update → resolve。

SPPM 的相机随机种子由像素索引和迭代索引共同生成：

$$
s=H(p\oplus k i),
$$

避免玻璃 Fresnel 分支跨迭代冻结。相机路径失败时保留已有有效 visible point，但每轮清零 `newFlux` 和 `newPhotonCount`。

方向约定为：相机射线从相机指向表面，而 BSDF 出射方向离开表面，因此 gather 使用 $\omega_o=-\omega_{\mathrm{camera}}$；光子方向指向表面，半球判定为

$$
\mathbf n\cdot(-\omega_{\mathrm{photon}})>0.
$$

反射、折射和漫反射延续均使用 $10^{-3}$ 法线偏移，并按新方向选择偏移符号，以减少自相交和错误遮挡。

## 5. 构建与运行

依赖 CMake 3.10+、C++17、CUDA 17 和 NVIDIA CUDA 工具链。推荐执行：

```bash
cd /workspace/FinalLab2021
cmake -S . -B build-cuda
cmake --build build-cuda -j2
```

```bash
./build-cuda/CudaPathTracing 32 64 64 8 diffuse mis /tmp/mis.ppm
./build-cuda/CudaPathTracing 32 64 64 8 diffuse sppm /tmp/sppm.ppm
./build-cuda/CudaPathTracing 1024 784 784 8 microfacet sppm /tmp/microfacet-sppm.ppm
./build-cuda/CudaValidation
```

## 6. 验证与限制

应检查 PPM 头部、图像尺寸、文件长度、输出字节范围 $[0,255]$，以及随迭代数增加的能量和颜色分布趋势。SPPM 中 `radiusSquared`、`photonCount` 和累计发射光子数应保持非负，参与 resolve 的值应保持有限。MIS 与 SPPM 在低样本数下不应逐像素比较，而应比较区域统计量、颜色趋势、边缘伪影和收敛行为。

当前已完成 CUDA 构建、64×64 SPPM 回归以及 MIS/SPPM PPM 格式检查。低迭代数下仍可能出现纯黑像素，不能单独据此判定算法错误；玻璃/镜面场景的高迭代收敛、显式相机侧半球检查、KD-tree 与线性 gather 对照，以及折射率权重的独立数值测试仍需继续验证。玻璃球反射左侧红墙的颜色由场景几何和镜面反射产生，属于合理的物理现象。

详细的 SPPM 开发、调试和后续工作记录见 [`SPPM_DEBUG_NOTES.md`](SPPM_DEBUG_NOTES.md)。

## 7. 文件索引

| 文件 | 内容 |
| --- | --- |
| `CudaPathTracing.cu` | 渲染器主程序与迭代调度 |
| `CudaKernels.cu` | GPU 求交、BSDF、MIS 和 SPPM 实现 |
| `CudaPhotonMap.cu` | 光子 KD-tree 构建与上传 |
| `CudaScene.cpp/.cu` | 场景数据转换与上传 |
| `CudaTypes.cuh` | 主机--设备共享结构 |
| `CudaValidation.cu` | CUDA 几何求交验证 |
| `SPPM_DEBUG_NOTES.md` | SPPM 调试记录 |
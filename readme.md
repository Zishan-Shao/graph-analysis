# DiT-XL/2 ImageNet-256 activation dataset

This repository implements the **class-conditioned DiT-XL/2** plan below. The
checkpoint accepts ImageNet-1K class IDs. We use a fixed subset of 10 classes
with five independent initial-noise seeds each, giving 50 generated images.
These are **class indices**, not indices of photographs in an ImageNet archive:
the generator starts from noise and does not read source images.

## Reproduce the dataset

Use Python with CUDA-enabled PyTorch and install `requirements.txt` in that
environment. These results used PyTorch `2.10.0+cu128`; the other direct
dependencies are pinned in `requirements.txt`. The scripts use the pinned first-party
[`facebook/DiT-XL-2-256`](https://huggingface.co/facebook/DiT-XL-2-256)
checkpoint and its DDIM scheduler (`eta=0`). The first sampling run downloads model
weights into `data/raw/hf-cache/`.

```bash
python -m pip install -r requirements.txt
python scripts/make_imagenet_conditions.py

# Smoke test: one image plus nine activation snapshots.
python scripts/sample_dit.py --limit 1

# Generate the remaining conditions on selected usable GPUs.
python scripts/run_sharded.py --gpus 1,4,6

# Check all 50 images and 450 trace files.
python scripts/validate_dataset.py
```

The committed `data/conditions/imagenet_10_classes_5_seeds.jsonl` contains the
exact class IDs, class names, and noise seeds. Each line is one condition, and
the zero-based manifest index is its line number minus one. A clone can sample
directly from this file without downloading ImageNet. `make_imagenet_conditions.py` recreates
it from the [pinned PyTorch ImageNet class list](https://raw.githubusercontent.com/pytorch/hub/c3beaae7d32fca2a23fec30aa7938ef5c9b6e5d5/imagenet_classes.txt)
and checks that list's SHA-256. The fixed, zero-based class indices are:

| Class ID | Class name | Manifest indices |
|---:|---|---:|
| 207 | golden retriever | 0–4 |
| 281 | tabby | 5–9 |
| 340 | zebra | 10–14 |
| 386 | African elephant | 15–19 |
| 402 | acoustic guitar | 20–24 |
| 404 | airliner | 25–29 |
| 620 | laptop | 30–34 |
| 779 | school bus | 35–39 |
| 817 | sports car | 40–44 |
| 963 | pizza | 45–49 |

Each five-row group has replicates `r00` through `r04`; the manifest records
their exact seeds. To regenerate one condition without running the others:

```bash
python scripts/sample_dit.py --sample-id imagenet-0207-r00 --overwrite
# Equivalent selection by zero-based manifest row:
python scripts/sample_dit.py --manifest-index 0 --overwrite
```

The seed for each row is the first four bytes, read as an unsigned big-endian
integer, of SHA-256 of `dit-xl2-256:2026:<class_id>:<replicate>`. The committed
manifest is the source of truth for sampling; its metadata file records the
class-list URL, SHA-256, selected IDs, and seed base. Exact pixels can vary
slightly across CUDA hardware or software versions.

Each run saves `image.png`, `metadata.json`, and nine compressed `.npz` traces
under `data/generated/dit_xl2_imagenet256/<sample_id>/`. A trace stores the
block input `h_in` and block update `residual = h_out - h_in` for all 256
spatial tokens. Calls 10, 25, and 40 and blocks 4, 14, and 24 are recorded.
Token index `i` maps to row `i // 16`, column `i % 16` on the 16×16 patch grid.
With classifier-free guidance scale 4.0, traces use the conditional branch.
The arrays are saved as FP16 after subtracting in FP32. Metadata records the
true diffusion timestep, model revision, scheduler, seed, and software versions.

Downloaded weights and generated arrays are ignored by Git; the code and
condition manifest are tracked. `run_sharded.py` assigns manifest rows by
stable index modulo the number of selected GPUs and writes worker logs under
`data/generated/dit_xl2_logs/`.

## Project analysis plan

**针对这份 E-1～E-7，我建议把设计收敛成：一个模型、一个 sampler、50 个生成样本，在不同层和采样阶段采集 token activation，完成一套图分析，最后加一个小规模的加速可行性实验。**

我前面建议“两种 sampler＋完整 50-step trajectory”，偏向后续研究的数据储备，**不是这次课程作业的必要规模**。这次更重要的是：数据定义清楚、图构造合理、分析完整，而且结论能为后续加速提供信息。

## 1. 项目研究什么？

建议题目就叫：

> **Graph Structure of Token Activations and Updates in Diffusion Transformers**

主问题是：

> **Diffusion token 的 activation 是否形成可解释的图结构？这些结构与 token 在当前 block 中的更新有什么关系？**

这样分工很清楚：**activation 是构图依据，block update 是用来解释图结构的属性，推理加速是 downstream motivation。** 不预设一定做 reuse、merge 或 FP8。

## 2. 数据集：我建议直接固定这个版本

| 项目 | 建议配置 |
|---|---|
| **模型** | DiT-XL/2，ImageNet-256；该设置为 256 个空间 token、hidden dimension 1152。[GitHub](https://github.com/facebookresearch/DiT/blob/main/models.py) |
| **Sampler** | DDIM，\(\eta=0\)，50 次 denoiser 调用；固定为确定性采样。[arXiv](https://arxiv.org/html/2010.02502v4?utm_source=chatgpt.com) |
| **生成样本** | 预先选定 10 个类别，每类 5 个独立初始噪声，共 50 条轨迹 |
| **采样位置** | 第 10、25、40 次 denoiser 调用，同时记录真实 diffusion timestep |
| **分析层** | 第 4、14、24 个 block，按 1-based 编号，覆盖浅层、中层、深层 |
| **保存内容** | Token 的 block 输入、block 更新、空间位置，以及生成配置和最终图像 |
| **特征维度** | 主实验 \(d=16\)，补充比较 \(d=8,32\) |

**针对课程作业，我会把之前的连续 13/14/15 层改成浅／中／深三层。** 因为这次首先需要观察结构随网络深度如何变化；连续层之间的相关性，可以留到后面的加速研究。

### 每个 datum 的定义

**一个 datum = 一条生成轨迹在某次调用、某个 block 中的一个空间 token observation。**

记录同一 block 的输入 \(h_i^{\mathrm{in}}\) 和输出 \(h_i^{\mathrm{out}}\)，定义更新 \(r_i=h_i^{\mathrm{out}}-h_i^{\mathrm{in}}\)。这里取的是整个 block 前后的 residual-stream 表示，不混用不同归一化位置；DiT 官方 block 本身就是在输入上依次加 attention 和 MLP 的更新。[GitHub](https://github.com/facebookresearch/DiT/blob/main/models.py)

按上述配置，共有 **450 个 graph snapshots、115,200 个 token observations**。但独立生成轨迹仍是 50 条，不能把所有 token 当作独立重复实验。

存储方面，可以保存 **FP16 的 \(H^{\mathrm{in}}\) 和 \(R\)** 两个数组，原始张量约 **506 MiB**，不含图片和元数据。建议采集时先用 FP32 计算差值 \(R\)，再转换为保存精度，而不是只留下两个 FP16 activation，之后才相减。

## 3. Graph experiment：严格按照 E-1～E-7 来做

**每个 sample／采样位置／block 单独构图，不把不同条件的 token 混成一个大图。**

### E-1、E-2：从数据到特征，再到相互作用矩阵

用 **block input activation** 做 PCA，得到 \(x_i=P_d(h_i^{\mathrm{in}})\)。建议每层用固定的参考样本拟合 PCA，不同采样位置沿用该层的投影；例如 10 条轨迹用于拟合与调试，另外 40 条用于主要分析。

然后用非负的 Gaussian affinity：

\[
S_{ij}=\exp\!\left(-\frac{\|x_i-x_j\|_2^2}{2\sigma^2}\right),\qquad S_{ii}=0.
\]

第一版可以把 \(\sigma\) 定为该图非零 pairwise distances 的中位数，扫描不同 \(k\) 时保持不变。Gaussian affinity 配合 kNN 是标准的相似度图构造方式。[Max Planck Institute](https://people.kyb.tuebingen.mpg.de/ule/publications/publication_downloads/Luxburg06_TR.pdf?utm_source=chatgpt.com)

**当前 block 的真实更新 \(r_i\) 不参与主图构造。** 它留到后面用于解释结构和检验预测价值。

### E-3～E-7：具体交付什么？

| 作业要求 | 我建议你们实际完成的内容 |
|---|---|
| **E-3：kNN 与连通分量** | 扫描 \(k=2,4,8,16,32,64\)。采用 union-kNN：任意一方把另一方选为邻居，就保留无向边。画 connected-component count 随 \(k\) 的变化。 |
| **E-4：组合结构** | 对大连通分量画 degree distribution 和 local clustering coefficient distribution。第一版这两项用无权邻接关系计算，描述实际形状，不预设它们服从某种分布。 |
| **E-5：谱分析** | 用保留下来的相似度作为边权，计算 normalized Laplacian；观察连通之后的 \(\lambda_2(k)\)，画前 25 个特征值、3D spectral embedding，并完成题目要求的 nodal-domain partition。 |
| **E-6：可选方法** | 用 k-means 作为额外 clustering baseline，比较它与谱划分的结果。**这里放聚类对照，不把 token repair 当作 E-6 的替代。** |
| **E-7：解释结果** | 将分组映射回图像 token 网格，比较不同组的更新大小、更新方向和空间分布，再比较浅／中／深层与早／中／晚采样位置。 |

谱分析统一使用 \(L_{\mathrm{sym}}=I-D^{-1/2}WD^{-1/2}\)，其中 \(W\) 是对称、非负的加权邻接矩阵。这与上面的 Gaussian＋无向 kNN 构图一致。[Columbia University Computer Science](https://www.cs.columbia.edu/~jebara/6772/papers/Luxburg07_tutorial.pdf?utm_source=chatgpt.com)

E-5 有一个细节要做好：**nodal domains 是特征向量同号节点形成的连通区域，不是简单地把所有正值节点归为一组、所有负值节点归为另一组。** 按题目要求寻找分离较好的非零简单特征值；没有找到时如实报告，不强行制造一个“很明显的 spectral gap”。

展示时也不需要放 450 张图：选一个**预先固定**的 sample 展示完整分析流程，再对其余样本汇总统计，避免只挑视觉效果最好的例子。

## 4. 加速怎么接？加一个小的“结构是否有用”实验

**这里我建议只做一个候选验证，不要求这次作业完成加速系统。**

最直接的候选是：

> **在只观察部分 token 的真实更新时，activation graph 是否有助于估计其余 token 的更新？**

例如，固定相同的 anchor tokens，分别观察 10%、20%、30% 的真实更新，比较：

| 方法 | 对未观察 token 的处理 |
|---|---|
| **不修正** | 预测更新为零，即保留 block 输入 |
| **全局修正** | 用已观察 anchor 的平均更新 |
| **图局部修正** | 利用图邻域关系和 anchor 更新进行估计 |

在未观察 token 上比较更新重建误差，画一张 **anchor fraction—reconstruction error** 曲线即可。测试样本不能用于选择最有利的参数。

这个实验的价值是：

**如果图局部修正有效，说明值得继续研究“少量精算＋局部修正”；如果无效，就说明当前图关系还不足以支持这个应用，而不是课程项目失败。**

也要明确：这是**离线重建实验**。它尚未证明少算部分 token 就能按比例省时间，也没有计入构图、选择和修正成本。真正的 kernel、视频质量、ZEUS 联合使用和多卡延迟，属于后续加速工作。

## 5. 这次不必做什么？

**不必把多模型、多 sampler、全部 timestep、视频模型和 distributed execution 同时放进来。** 它们会扩大工作量，却不是老师这份要求的核心。

主实验完成后，有余力再增加同一模型下的一种 sampler，作为补充稳健性检查；没有增加，就把结论明确限定在当前 checkpoint 和采样配置，不声称跨模型普适。

最后，给老师的项目描述可以直接写成：

> 我们构建 diffusion Transformer 的 token-level activation dataset，研究其在不同网络层和采样阶段的相似度图结构。通过 kNN 连通性、局部聚类和 Laplacian 谱分析，我们检验这些结构与 block 更新之间的关系，并初步探索它们能否支持选择性计算与局部更新预测。

**我认为这就是合理的规模：一个受控的数据集、一套完整的课程分析、一个与加速有关的小验证。先把这三件事连起来，比先收集很多模型和 sampler 更有价值。**

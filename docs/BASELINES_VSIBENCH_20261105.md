# VSI-Bench baseline 核验与 Spatial-135K 适配

2026-11-05。工作目录 `/home/u2024311136/SandMem`。用户本轮明确优先此项，暂缓 StreamVGGT；随后确认全部 baseline 使用 Spatial-MLLM 135K，接受独立适配。没有修改 SandMem 方法、Proposal、AGENTS 或 PLAN。

**最新范围：用户确认本轮先完成 HERMES/CausalMem，暂不为 ReKV/FlexMem 引入新位置规则；指定 ReMem 未找到公开实现，不纳入本轮 baseline。** HERMES/CausalMem 全量作业均已正常结束，分别为 47.60077734 / 47.09578305 分（完整 VSI-Bench、Spatial-135K 独立适配版），结果及成本见[完成报告](BASELINES_VSIBENCH_RESULTS_20261105.md)。下方接线/排队段落保留过程记录，以最终完成报告为当前状态。最终核对确认所有源帧/2D处理像素/网格相同，1881块中1块融合特征存在数值差异，已单列，不能声称全量逐位一致。

## 论文与代码核验

| 方法 | 指定论文 | 作者代码 | VSI-Bench 可引用分数 |
|---|---|---|---|
| ReKV | https://arxiv.org/abs/2503.00540 | https://github.com/Becomebright/ReKV | 指定论文全文未检出 VSI；本轮跨论文搜索未找到可确认分数 |
| FlexMem | https://arxiv.org/abs/2603.29252 | https://github.com/city1517/FlexMem | 同上 |
| HERMES | https://arxiv.org/abs/2601.14724 | https://github.com/haowei-freesky/HERMES | PDF 全文未检出 VSI；搜索命中的 44.6 是 OVO-S-Bench，59.20 是 OVO-Bench，不可移用 |
| CausalMem | https://arxiv.org/abs/2606.25658 | https://github.com/hktk07/CausalMem | 指定论文全文未检出 VSI；本轮跨论文搜索未找到可确认分数 |
| ReMem SCM/RVM | https://arxiv.org/abs/2610.05940 | 尚未找到可确认的作者实现 | 同上；不能用其他同名 ReMem 代替 |

上述为“未找到”，不是证明所有后续论文均未测过。原文 HTML、HERMES PDF、内容提取及 SHA256、四个官方代码库和评测库的 commit 已保存在 `work/baselines_vsibench_20261105/sources/` 与 `reports/source_audit.json`。GitHub 检索命中的 `jinlab-imvr/ReMem` 对应另一篇 *Reasoning with Memory: A Temporal Granularity-Adaptive Framework for Training-Free Long Video Understanding*，不对应用户指定的 SCM/RVM 论文。

本地旧 `/home/u2024311136/ReKV` 有未提交修改；其 prepare_vsi.py 实际适配 VSI-SUPER。本次保留原目录，使用独立官方 clone，不将旧结果视作 VSI-Bench 正式结果。

## 数据与评分

- 官方数据 https://huggingface.co/datasets/nyu-visionx/VSI-Bench ，固定 revision `3a6a1cf9ac54c403509b15260897cc6ddc41b408`，读取 `test.jsonl`，实际 5130 题、288 视频、10 个原始 question_type。
- 原本地缓存仅 `test_debiased.parquet`，不能代替完整集。下载三份原始视频 archive，逐一验证官方 LFS SHA256，并保存解包回执。数据与结果均在独立 work 目录。
- 官方评分代码 https://github.com/vision-x-nyu/thinking-in-space ，固定 commit `51e089c3ae69b9435e9489058610f5b3964c56a8`。
- 使用官方 fuzzy_matching、MCA exact_match、NA MRA 和 aggregate 的全部运算，包括官方 MRA 的实际阈值实现。相对方向三难度先取平均，再计算八类任务平均。不能改成所有问题的微平均。CPU 检查发现官方 aggregate 对较旧 pandas 返回的 Python float 调用 `.item()` 会报错；兼容层只把两处标量 `.item()` 转成 `float()`，不改任何评分或聚合运算，并在 scorer 回执中披露。
- 推理清单删除 ground_truth；按 question id 连接评分。正式评分严格要求完整 5130 个唯一 id；工程失败保留原始失败，预测为空并计入完整分母，不丢题。full 和 debiased 不混称。

## 共享 backbone 与接线验收

冻结 `Diankun/Spatial-MLLM-v1.1-Instruct-135K`，revision `635b63733ed24f76b2f1a533dd5bc32836ea6c53`。沿用已完成加载检查的独立 BF16 环境、官方 processor 与双路视觉编码器。全部权重冻结，禁止量化、换 checkpoint、绕开 Spatial 空间流。

原生 baseline 并无此 checkpoint 的官方结果；本次新结果必须标为 **Spatial-135K 适配版**。这些原本是 2D 记忆方法，但共享 Spatial backbone 自带空间编码器，不能称整个适配系统完全没有 3D 模型。SandMem 的外部几何/沙盒流程不参与 baseline。

当前首先实现两条接线：

1. CausalMem：从固定作者文件原样编译 FOSSCache 定义，保留在线语义基更新和 token 选择。输入改为 Spatial 实际双路融合后的 token；按保留 token 的原始位置携带 M-RoPE 坐标。offline 默认 budget=12000、k_max=64、decay=.9、max_new_basis=8、time_weight=0；不训练拟合参数。
2. HERMES：原样加载作者的分层保留、三种伪问题、跨层分数和位置重排定义。沿用作者 load_model 的 .1/.3 分层比例和 6000 KV 额度。适配 Spatial decoder 的命名及 transformers 4.51 预计算 RoPE 接口，给每层同步 position_ids 和 position_embeddings；避免只修改已不再生效的参数。

ReKV/FlexMem 尚需适配原 Qwen2 注意力/一维 RoPE 与 Spatial M-RoPE 的差异；不能直接套旧 patch 后宣称保留 Spatial backbone。FlexMem 的 MemEncode 与带拟合步骤的 MemIndex 必须区分；尚未选择/跑出正式适配结果。

接线 pilot 固定使用 Spatial 官方资产中的 VSI-Bench 视频 arkitscenes/41069025，按输入字段哈希各选一个数值题和选择题（id 167、1236），选择时不读取答案。完整视频 1 FPS、连续 16 帧一块，最后不足块仅由原生 processor 做时间 patch padding；不按题型选择采样或记忆机制。使用原生视觉预处理默认值，保存实际网格/像素张量摘要/源帧映射；不增加帧数上限或暗中截断。

pilot 阶段先检查共享视觉编码接口、原生 official forward 与提取的融合特征逐元素一致、源帧映射、真实记忆压缩和输出，之后已冻结正式预算。各方法保留必要流程；视觉 token、KV/特征内存、文字、输出、伪问题和额外调用分别计费，不称严格等预算。共享融合特征缓存只可在输入、processor、checkpoint、chunk 划分全部一致并校验后复用；本次两个正式运行各自完整编码、没有跨方法复用融合特征。

pilot 输出使用官方 VSI-Bench 的 16 token、greedy；不按 pilot 答案改提示、参数或预算。先只做工程验收，不计算 pilot 正确率。完整比较需先通过接线，再冻结所有源码、数据清单、配置与调用额度。

## 接线阶段历史状态

尚无本次 Spatial baseline 的 VSI-Bench 分数。CausalMem 接线作业 17270 已完成（COMPLETED/0:0、12:36），A100-PCIE-40GB；输出 `work/baselines_vsibench_20261105/pilot_01/`。共 169 个真实采样帧、33660 个融合 token，经原生 FOSSCache 保留 12000 token；两题输出分别为可解析数值和选择字母，不核对答案。官方融合特征逐元素一致通过，峰值分配 13.25 GiB、保留 14.22 GiB。接线总程序时间 558.95 秒，包含加载/独立等价性对照/重复编码；不冒充纯推理耗时。HERMES 单独接线作业 17271 已提交。一次额外提交被账户单作业上限拒绝，未产生模型调用。

下载首次在全部字节到齐后遇到系统 Python 3.8 缺少 `hashlib.file_digest`；原失败保留在 `reports/download.log`，改用流式 SHA256 后直接复用完整 `.part`、不重复下载，三个包均已校验解包，见 `download_02.log`。评分首轮标量类型错误保留 `scoring_check.log`，兼容后的验收输出另存 `_02`。

## 正式运行接续

HERMES 的同输入接线 17271 也已完成：两次回答，实际进行了分层压缩与缓存回收，原生融合特征逐元素一致；没有用答案评价或改动设置。

完整视频元数据已逐条读取：288 视频、累计 27812.4833 秒，1 FPS 共 27946 个真实采样帧，16 帧编码共 1881 块。单方法正式上界 1881 次视觉塔、1881 次空间塔、1881 次融合、5130 次回答、82080 输出 token；HERMES 另有每块三种伪注意力查询，共 5643 组，不能藏入普通 QA 成本。所有样本/帧均按输入结构固定，不按答案选取。

- CausalMem 正式目录 `work/baselines_vsibench_20261105/formal_causalmem_03/`：封存 `manifest.json`、`config.json`、`budget.json`、5130 题无答案 `inputs.jsonl`、依赖哈希和完整 runtime。A100 作业 **17272** 已启动。
- HERMES 正式目录 `work/baselines_vsibench_20261105/formal_hermes_01/`：相同清单和共享输入协议已封存，串行控制器等待 CausalMem 结束后提交。**等待控制器不等于已提交 Slurm，不填 HERMES 正式作业号或成绩。**
- 每个作业单 A100、单 CPU、12 小时；只在视频边界保存并继续未尝试视频。已记录失败不自动重答；连续两个视频工程失败即停止。串行控制器不根据分数采取任何分支，最多八个作业只是工程终止保护，不扩大固定数据/模型调用额度。
- 每视频保存源视频哈希、真实源帧序列、实际网格与像素摘要、实际融合特征 `.pt`、预算与处理时间、全部原始回答/失败；HERMES 另保存实际每层 KV 长度及回答 token，CausalMem 保存全部保留 token 的原始位置及实际输入文字 token。完成全部 5130 题后才调用官方评分，在对应目录输出 `predictions.jsonl`、`scores.json`。部分进度不能称完整成绩。
- CausalMem 在正式前补充了两个纯工程约束：按视频名称哈希重置随机种子，保证跨作业续跑不改变在线 SVD 的随机流；仅保存保留 token 的位置坐标，避免辅助位置表无界增长。CPU 测试在不同网格/多次压缩下证明有界表与完整原始坐标逐元素相同（`causal_positions_check.log`），不新增 QA 或调整选择规则。早期 `formal_causalmem_01/`、`_02/` 仅预封存、0 模型调用，完整保留；唯一正式执行目录为 `_03/`。

可复现入口：`scripts/baselines_vsibench/freeze.py`、冻结目录内 `runtime/full.py`、`full.sbatch` 和 `serial_controller.json` 的实际提交命令。下载入口 `download_data.py`，CPU 评分入口 `score.py`，评分验收 `check_scoring.py`。`scripts/baselines_vsibench/` 均为独立适配，不修改 SandMem 核心流程。

ReMem 已追加核对指定 PDF 外部链接和 GitHub 论文题名/编号/关键词检索，仍未找到相应作者实现；检索回执在 sources。已集中询问用户是否持有代码入口，其余项目不因此停工。ReKV/FlexMem 仍处于 Spatial 注意力与位置缓存适配阶段，尚无 GPU 正式调用或分数，不能把 HERMES/CausalMem 的接线成功推广到它们。

ReKV/FlexMem 的实际源码还暴露一个实验定义空缺：前者 RotaryEmbeddingESM、后者 Memory._step/Qwen2RotaryEmbedding 都使用一维缓存重排，没有 Spatial 时间/高/宽三轴的官方规则。不能悄悄将 Spatial 的空间位置塌缩成一维后称为公平移植。集中澄清后用户选择 **先完成 HERMES/CausalMem，暂不为另外两项引入新位置规则**；因此不继续移植或提交 ReKV/FlexMem。ReMem 根据用户决定从本轮 baseline 清单排除，未找到公开代码不等于断言作者永远不会发布。

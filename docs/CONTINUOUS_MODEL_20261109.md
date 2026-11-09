# SandMem 连续几何修正：实际模型入口

用户要求按 coauthor 的 Scene 分组方案修正实际模型。本次已将连续几何、实例记忆、Scene、3D渲染、新选帧与实际 QA 调用接成独立可执行入口；不再只有几何探针。旧 D116 源码/配置/冻结产物及所有旧失败保持。当前新增模型是未重新冻结的修正版，不把旧得分改标签使用。

## 实际入口

- `sandbox_vlm.continuous_model.ContinuousSandMem`：实例身份、观察、全采样帧描述子/相机、Scene、容量处理与 QA。
- `sandbox_vlm.perception.continuous_backend.ContinuousPerception`：真实 StreamVGGT 连续 FIFO、冻结 SAM 图像特征/实际区域 mask、同一冻结 MLLM 发现。
- `scripts/run_continuous_model.py`：原视频到实际模型回答，无旧实例/预测或答案输入。
- `configs/continuous_model_20261109.json`：显式采样/几何/FIFO/发现/容量/QA 额度。
- 独立发布目录 `~/SandMem/` 新增相同实现与 `create_continuous_model` API；旧 `create_memory`/`compose` 等保留为冻结 D116 兼容接口。

所有采样帧按时间顺序进入 StreamVGGT，不由检索决定几何输入。每帧冻结 SAM 图像编码提供视觉描述子；只在已声明发现帧调用 MLLM，并对成功发现的实际对象做 SAM 框提示。读取 SAM 原图像编码器 hook 输出，发生在 no-memory embedding/时序条件化之前；实例区域平均池化与帧级平均池化共享这一次编码，不添加外观模型。当前参考入口的发现帧是声明额度内按时间均匀选取，未新增问题相关性筛选模型；调用者可以注入现有发现日程，但不能改变完整几何帧清单。

同 caption（大小写/空白规范化的完整名称）内按冻结外观余弦阈值做同帧联合一对一分配，带未匹配虚列；未匹配且有实际 SAM 证据才新建 ID。配置阈值初值0.90，不用跨帧2D IoU、Scene ID或3D框作身份门槛，不添加 additional correspondences，不维护待定身份池。匹配更新最近实际外观，保留首次存入的名称/完整描述；不引入描述改写模型。

可靠世界点图直接用于 Scene 与 mask 内可见点 AABB，不二次施加相机变换。相机解码为 world-to-camera 后仅取逆一次。物体框为同一实例/参照下最早仍存活的可靠观察的静态可见表面代理，不是多视角完整体积。Scene 是连续局部观察组，按当前体素重叠与初始固定对角线上限划分；FIFO 不切 Scene，不重置相机参照。

实际渲染在同一参照坐标中显示 Scene 范围、相关实例框和连续相机轨迹，标注被选中的真实帧。不同参照不得直接比较几何；当前真实连续入口只处理一个明确上游参照，遇到显式独立重建需开始新会话，不自动假装建立坐标连接。

QA 先从完整可见采样序列选时间种子，再补相关实例覆盖，最后视觉去冗余。真实帧从源图像按ID读出，沙盒图作为独立额外图片实际入模；原问题及所选记录的完整字段不改写。稀疏输入且存在实际区域/几何链接时执行独立读帧候选→联合核验；其余联合一次。保留 D065 基础答题首段及两句坐标/来源说明，核验引用仍只认证所提供帧号，不认证视觉判断。调用者显式传入原输入结构的 sparse 标志，不能按测试题型或得分选择分支。

## 容量与来源

默认64MiB，对实例、观察、全部帧描述子/相机/场景归属、活动体素、Scene、配置等可达状态及两份序列化包络共同计费。先精简每实例观察为最早与最近，再淘汰最久未观察实例；清理所有 Scene/帧引用，ID不复用。

物体框的 source_observation_id 若被压缩删除，同步删框，不额外钉住中间观察、不从旧归档恢复；后续实际可靠观察可重新建立框。框及来源在完整入模记录中明确列出。不可压缩的帧/Scene必需状态仍超限时明确失败，不用磁盘分页掩盖。模型权重、密集几何工作区及KV与外部结构预算分开；这不是进程64MiB认证。

## 可执行用法

从仓库目录运行（环境须含固定参考 Qwen/SAM 依赖；本入口不自动迁移 Spatial）：

```bash
python scripts/prepare_continuous_source.py --output /external/runtime/streamvggt
# 准备配置中固定 revision 的 StreamVGGT model.safetensors，校验官方SHA。
export SANDMEM_PYTHON=/path/to/python
sbatch scripts/run_continuous_model.sbatch \
  --config configs/continuous_model_20261109.json \
  --backbone-config configs/reference_qwen.json \
  --upstream /external/runtime/streamvggt \
  --checkpoint /external/weights/model.safetensors \
  --video /path/video.mp4 --question 'Original question' \
  --output /external/results/new_run --sparse --answer-format text
```

原工作区可把 backbone config 换为 `configs/vsi_super_d065_spatial.yaml`。不带 `--max-frames` 时遍历完整采样序列；`--max-frames 64` 仅用于前缀工程验证，不能评分为原完整视频任务。总真实帧额度由 `qa.real_frame_budget` 传入，与图合计预算须一致；稀疏/密集输出256/64，结构容量64MiB。当前配置的8发现帧、1536发现输出、8真实QA帧是接线初值，与D116历史发现512等额度不同，不作为公平性能对照。

`--answer-format choice` 只验证最终是否A–D；`number`只验证有限数，均不求解答案、不改变分支、不触发重答。自由文本默认text。入口拒绝覆盖结果目录，并记录调用尝试、逐帧几何/发现失败、内存、实际消息/图像像素hash和模型原始输出。

## 验证与真实失败

27项CPU测试通过，覆盖一对一分配的贪心冲突反例、跨帧框位置改变、未匹配新建、ID不复用、首末压缩与中间几何来源清理、最久未观察实例淘汰、不可压缩预算失败、实际3D图/中间帧进入答题消息、两阶段图像对象身份保持、输出合同拒绝且不重试等。

真实端到端作业17809：A30，COMPLETED/0:0，Slurm2:17，模型流程123.98秒；原开发记录9的连续原帧0–63，64次StreamVGGT/8次新Qwen发现/64次SAM图像编码/30次实际SAM提示/2次QA，0历史语义记忆或预测复用、0重试。

- 最终30个实例记录、28个Scene，均为估计记录数量，不是真实物体/房间数量。
- 选择原帧 `[0,9,18,21,42,45,54,63]`；F21/F42无实例观察记录，通过时间初始化实际入模。
- 第二遍实际输入9张图片：新3D图+8真实帧，2064视觉token。相关23条完整记录和23个有实际来源的框逐一核验。
- 130份封存源码/配置/计划hash、真实图像像素hash及帧顺序审计通过；首末记录不被QA改写。
- 当时结构测量1,661,452字节（后续将配置等完整对象也纳入保守计费，只有计费更严格）；CUDA峰值allocated13,530,411,008/reserved15,084,814,336字节。
- 回答为“None of the options are correct.”，不是原题要求的A–D。原始运行completion保存了解析层的ok，独立input_audit明确将该回答判为格式失败；原输出不覆盖。当前模型已增加外部答案合同验证，拒绝该答复，不重试、不挑另一阶段答案。

这里截取64帧，仅验证接线；原完整视频613帧的问题没有被拿来报告准确率。新代码未取得新全量或完整视频效果分数，不复用D116成绩。实际生成文件位于原工作区 `work/bench/continuous_model_wiring_20261109_01/`。后续只加了输入checkpoint SHA校验、完整对象结构计费和最终答案合同拒绝，未更改这次归档及预测，也未重跑模型。

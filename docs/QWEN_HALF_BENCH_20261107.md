# Qwen D116：VSI-Bench与MindCube按类50%抽样

用户最新授权优先原Qwen3-VL-2B，在两个数据集按每类50%抽样测评。
Spatial 17522已在当前题边界结束，两个worker退出0；原结果和全部资源账保留，原续跑控制器停止。Vgent不改。

- Backbone：Qwen/Qwen3-VL-2B-Instruct，revision 89644892e4d85e24eaac8bacfd4f463576704203；所有筛帧、发现、QA实际调用均为冻结原Qwen，不复用Spatial或历史Qwen语义记忆。
- 方法：D116 + first/last compaction + oldest-observed eviction；完整相关记录、实际观察链接、局部几何和实际沙盒图/原帧、既有两阶段门控保持。外部SAM/VGGT不替换。
- VSI-Bench：原5130题，按原始question_type的10个标签各取ceil(N/2)，2567题。评分仍为固定官方八任务聚合（方向子类按官方口径合并）与数值MRA；抽样成绩明确不能作为全5130题成绩。
- MindCube：官方当前评测入口的tinybench，1050题，不使用10000题训练集或混合21154题总文件。按(setting,type)九个子类各取ceil(N/2)，527题：Among301、Around126、Rotation100。官方整体准确率及分组，错误/无效答案计错。
- 抽样：固定种子字符串 `SandMem-Qwen-D116-half-v1-20261107` 加白名单输入的SHA256排序；答案不参与排序，题型仅用于CPU分层，不传给推理。完整抽样清单在manifest.json封存，不按中途成绩改设置。
- MindCube源：MLL-Lab/MindCube，revision 9c941b46a6bd65b6914669ef7a579948fc9c8467；data.zip SHA256 714c8961d66e9662c826d2202493a7d86cbe31797cf41e3a0567cb4d72b0aaa8。官方代码commit b8b7062adf6d3e49d588a7d014a0a787553d09ec。来源：https://github.com/mll-lab-nu/MindCube ，https://huggingface.co/datasets/MLL-Lab/MindCube 。
- MindCube只有2–4张有序视图，使用独立原图读取器直接读取，不进行视频转码；不添加/插值视图，核验每张原图内容hash，并测试奇数边长图像逐像素不变。序号只是视图索引，不声称为物理时间。metadata、meta_info、真值地图/姿态和答案不入模；原问题内提供的视角说明保留。
- 原Qwen提示和解析/重复停止保持，不采用Spatial新增JSON语法、字段长度或processor对齐策略。发现512，稀疏/密集QA256/64，联合图像33/视觉8192/文字32768、结构64MiB；这些为D116实际联合证据额度，旧reference配置默认额度由D116接线覆盖时明确记账。
- 独立Qwen冷启动入口从已验收冷链路派生，只翻译模型特定的来源核验，不改原冻结147个文件。CPU AST契约测试验证方法函数的差异白名单，Qwen实际视觉编码器hook检查QA所有图像网格。
- 两个数据集各前2条固定输入作为接线gate，0成绩判据；通过后自动继续，完成才评分。每题由目录原子claim分配给一个GPU，0模型重试；未知工程错误暂停，失败原始输出保留。

运行：

```bash
/home/u2024311136/sandbox/venv/bin/python -B scripts/run_qwen_half_bench.py prepare --output <新的独立目录>
/home/u2024311136/sandbox/venv/bin/python -B scripts/run_qwen_half_bench.py control --output <已冻结目录>
```

首个CPU准备目录 `_01` 因OpenCV视频写出未通过逐像素校验而关闭，0模型调用，失败记录保留；`_02`因缺少FFmpeg命令行工具在CPU准备阶段退出，0模型调用。最终 `_03`采用原图读取器，不依赖视频编码，不接受错误像素。运行环境使用原Qwen venv，transformers4.57.6、torch2.5.1+cu124；双A30每卡单独进程，硬件和实际预算独立披露。

Proposal、AGENTS.md、PLAN.md均未修改。正式分数尚未产生，不预填。

运行回执：固定来源157文件、3094输入，5项CPU契约测试通过；`work/qwen_d116_half_20261107_03/` 已冻结并启动17529双A30，实际两worker分别可见设备0和1。Spatial17522在边界正常退出0/0，完成46题，保存`paused_for_qwen_20261107.json`。Qwen接线gate正在执行，尚无正式分数。

17529实际接线gate已通过：2933、3984、among_group605_q0_2_1、among_group412_q0_5_2均为完整Qwen感知、实际沙盒入模且有效解析，未读取成绩；自动正式抽样续跑已开启。该时点完成数 {"VSI-Bench": 5, "MindCube-tinybench": 5}，只报告进度，不计算中途准确率。

## 用户要求改用A100

新增独立硬件交接器 `scripts/run_qwen_a100_handoff.py` / `scripts/run_qwen_half_a100.sbatch`，3项交接测试通过。原冻结157文件、manifest、模型、提示、全部预算和3094样本保持，不编辑运行中的源文件。

实际调度限制：A100分区每用户MaxSubmitJobs=1，Vgent17432占位；每作业最多1张A100，不能预设一A100整体快于双A30。已对Vgent设置用户hold：它仍运行，仅使下次正常视频边界重排队后保持挂起；不终止任何当前推理。交接器等待已有约10小时正常执行边界，核验全部视频closed后才取消挂起的空作业释放提交额度，先确保A100提交，再让Qwen双A30在题目边界退出0/0，核验所有claim都有结果，继承全部已完成预测，以原worker单A100续跑。等待期间双A30不停止、不空耗等待。

授权与可复核交接状态在 `work/qwen_d116_half_20261107_03/a100_handoff/`，控制器PID570928。实际A100尚未分配，不能宣称已经切换。Qwen完成后用原未改Vgent launcher和保存进度重新分配作业继续。计费将同时披露之前A30及后续A100分配时间，不作为纯同硬件结果。Slurm运行中hold语义已查官方手册：https://slurm.schedmd.com/scontrol.html#OPT_hold 。Proposal、AGENTS.md、PLAN.md未修改。

## 最新用户指令：立即取消Vgent、A100优先完成MindCube

已按明确指令立即取消17432（Slurm CANCELLED，运行5:07:47），完整保留169个已完成视频和本次未完成scannet_scene0207_02的视频目录；该视频19个问题output已落盘、第20个问题已开始且可能有部分模型调用。具体完成视频数/原始输出以`user_cancelled_17432_20261107.json`为准；禁止将旧completion.json当成本次成功结束，禁止自动重试或自动恢复Vgent。此前等待正常边界的交接控制器570928已停止，旧计划已标注superseded。

新独立优先入口`run_qwen_mindcube_priority.py`直接调用原冻结`qwen_d116_cold.fresh_question`，保留原157来源hash、checkpoint、配置、全部输入和输出预算；0方法/提示/阈值改变。66条已完成MindCube预测原样复用，其余461条通过目录原子预留给A100，原A30仅继续VSI-Bench。两个dataset无问题重跑或跨数据集记忆共享。

A100作业17555已于17:48分配gpu1，单卡、2 CPU；先读取实际显存容量，根据已完成完整MindCube题测得峰值8.289GiB，加每worker4GiB和全局2GiB余量决定是否允许两个独立进程共享单卡，否则单进程。并发数仅为执行调度，不改输入、模型精度、采样或生成；测量不保证未见输入绝无峰值变化，OOM将显式失败，不截断或偷偷重试。硬件实际见priority目录`hardware_17555.json`。

该dataset全部527条完成后立即独立调用固定官方MindCube BasicEvaluator，不等待VSI-Bench；保存`scores.json`、`RESULTS.md`及根`mindcube_scores.json`。原A30题与新增A100题分开披露硬件与成本，不称纯A100结果或1050题全量成绩；无效计错，固定完整527分母。2项队列/类别隔离测试通过。Proposal、AGENTS、PLAN未改。

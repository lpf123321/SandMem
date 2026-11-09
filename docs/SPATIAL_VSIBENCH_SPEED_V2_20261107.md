# D116 等价工程加速第二版（2026-11-07）

用户最新选择是保持原D116，仅做等价工程加速，接受无法保证一天。此次没有启用按视频共享实例记忆，没有改成问题无关发现，没有缩减帧数、观察、token或几何预算。问题相关的筛帧、发现、实例关联、局部3D、真实沙盒图/源帧、相关完整记录及两阶段QA均沿用原单题入口。

## CPU验收

独立模块：`sandbox_vlm/backbones/spatial_fast_syntax_v2.py`。旧语法模块及正在运行的`_04`源码不改。

1. 原字符解析器的栈状态不可变；缓存同一状态的当前字段/顶层字符串查询，字符转移继续调用原JSON解析器。
2. 自由字段中无转义的原始控制字符始终不在既有合法字符集中，在词表路径入口排除这些必然非法的头部；引号、反斜杠及其后续转义路径继续经过原解析器。没有改动合法输出语言、box/词数规则、闭合规则、生成或严格解析。

复用原固定四帧16/89/477/729的实际token轨迹，包含两条历史超限失败；0新GPU调用、0新感知/QA、0评分、0补答。全部1978步完整合法token集合与现行v1及更早原版完全相同，mask文件SHA256为`9f0376f6422e45a1c29912fc290e0021829be4336e2216433ece2ba335a12d15`。

本轮相同CPU剖析口径：v1为227.1071秒，v2为197.3181秒，时间减少13.1167%，速度1.1510倍。计时包含cProfile，不是端到端GPU加速比，不能直接用于推算全量工期。栈查询、字段顺序、转义、中文/空白、八词边界、重复/非法框、闭合预算、两阶段/身份/图像绑定、5130输入分片和GPU隔离等55项检查通过（包括8项交接回归）。

证据目录：`work/spatial_speed_20261107_02/`；`parser_equivalence.json`绑定源码hash，`full_v1/`和`full_v2/`含逐步mask及剖析结果。验证只覆盖已保存轨迹与边界测试，不声称穷尽所有未来生成前缀或做了新的GPU数值复跑。

## 续跑与资源

新入口：`scripts/run_spatial_vsibench_parallel_v2.py` / `.sbatch`，单题仍调用原`run_spatial_vsibench.run`，两worker分别绑定一张GPU，原输入序号modulo 2分配，各2565题。旧`_04`已完成结果仅按来源及hash引用，不重跑、不改预测；失败也保留。新目录为`work/sandmem_vsibench_spatial_full_20261107_05/`，旧目录、源码、作业和全部资源消耗完整保留。

控制器核验旧调度器PID/命令后退休旧调度器，并用既有SIGUSR1通知17521在各自当前题边界停止；等待整个旧分配结束、检查无未完成输入后，再冻结新manifest、配置、源码及CPU验收回执并提交新作业。未知工程失败停止续跑，不重试发生过的模型调用，不静默改成评分错误。当前控制状态/日志：`work/spatial_speed_20261107_02/parallel_controller.json`、`controller.log`、`controller_pid.json`。

调度干跑另行确认：A100分区每作业只允许1 GPU，A30分区允许2 GPU；总QoS额度不能替代每作业限制。因此本版本继续使用两张独立A30，不实施此前提出的双A100自动切换、不共享同一GPU塞入两个进程。A100干跑失败回执保存在`work/spatial_speed_20261107_02/a100_dryrun.json`，0分配/0模型调用；Vgent仍正常使用其A100。每作业2 GPU/2 CPU、12小时调度与原9小时题目边界规则保持，实际型号/UUID/显存及分区均记录。

所有5130个结果齐备后才合并并调用原官方评分器。总费用须合并原接线/失败/旧正式作业以及v2分配；双卡分配GPU时间是分配墙钟时间的2倍。当前无新的全量分数，不承诺24小时完成。

Proposal、AGENTS.md、PLAN.md均未修改。

## 验收入口

```bash
work/envs/spatial135k/bin/python -B scripts/profile_spatial_discovery_speed_v2.py --output <新的CPU回放目录>
```

正式控制器已启动，不要重复提交同一续跑：

```bash
work/envs/spatial135k/bin/python -B scripts/run_spatial_vsibench_parallel_v2.py control \
  --parent work/sandmem_vsibench_spatial_full_20261107_04 \
  --output work/sandmem_vsibench_spatial_full_20261107_05 \
  --proof work/spatial_speed_20261107_02/parser_equivalence.json \
  --state-dir work/spatial_speed_20261107_02
```

## 17521交接退出异常核验

原双卡作业退出码为[-10,0]，-10是SIGUSR1终止。两个worker均已写出完成回执（19+17=36题），35号输入正常OK，无未完成目录、无shard推理异常。重启交接控制器期间重发过SIGUSR1；它与退出过程存在竞态，精确的Slurm/父进程信号投递先后未证明。原failure、allocation、日志和全部模型输出保留；CPU核验36个结果及worker/分配回执hash后才允许引用，不重跑。

v2将转发改为每worker至多一次，跳过已完成回执的进程；初始化阶段暂存信号并在原core安装处理器后传递，原core返回后忽略SIGUSR1。负例回归明确拒绝SIGKILL、真实推理异常、缺结果、回执计数错误及Qwen来源。此次核验和回归0新增模型调用；已核验交接回执见`work/spatial_speed_20261107_02/parent_shutdown_reconciliation.json`。

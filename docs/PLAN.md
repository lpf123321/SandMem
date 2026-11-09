# Fixed-method release

- Fixed: D116 inference/selection/rendering plus the accepted instance-memory capacity policy.
- Verify: exact frozen-source hashes, independent-checkout CPU regressions, and optional 294-query D116 input equivalence using external artifacts.
- Separate work: Spatial-MLLM production migration, checkpoint/baseline/budget confirmation, formal comparisons and ablations.
- Do not relabel old Qwen scores as new capacity or Spatial results.


## 2026-11-09 continuous model correction

The user explicitly requested implementation of the coauthor Scene scheme in the actual model, superseding the old no-method-change restriction for this isolated candidate. Current entry is scripts/run_continuous_model.py / api.create_continuous_model. Original D116 files, aliases and results remain frozen. The candidate adds continuous StreamVGGT FIFO, current-Scene overlap/fixed extent, same-caption joint one-to-one appearance assignment without cross-frame 2D IoU or extra geometry associations, source-backed visible-surface boxes, source-clean capacity compaction, actual shared-reference 3D rendering and temporal/instance/diversity QA. No pending pool or count solver. See docs/CONTINUOUS_MODEL_20261109.md for exact budgets and limitations.

27 CPU contract tests and one A30 64-frame Qwen-reference end-to-end input check completed in the original workspace. Real QA received new 3D and middle frames F21/F42. The prefix response was outside A-D and is preserved as an answer-format failure; no score claim, no retries. Do not relabel this as full-video, Spatial, identity-accuracy or new frozen result. Existing ongoing evaluation work and all uncommitted files are protected. These AGENTS/PLAN additions must be reported.


## 2026-11-09：测试前审查，正式评测不放行

用户要求先检查修正版能否重新测试，避免全量完成后才发现工程错误。本次修复逐阶段答案合同/同实例同参照两阶段门槛/选项污染、发现失败即停、完整与截断标识、输入预算继承及CPU启动检查。长视频改为损失为零的逐条快照，全部可达状态+256KiB工作区计费，不再同时展开两份全量浮点JSON；仍保留全部源帧、原精度和首末/最久未观察实例容量规则。

38项CPU测试通过；32真帧+1图的原生processor检查为33图/4656视觉token（0GPU前向）。14,400帧/每帧1Scene合成元数据状态约46.8MiB（含最终256KiB序列化工作区），仅为压力状态而非真实长时GPU验收。

单完整开发视频门槛17850在A30执行，原记录9的613帧未截断，源帧165第18次发现输出重复而停止（304token，未到1536上限）；166几何/18发现/165SAM编码/64提示/0 QA，FAILED/1:0、4:48，0重试，原始重复文本与全部失败保留。没有运行批量评测或评分，不改提示/格式/阈值搜索来跳过失败。

因此正式全量仍禁止放行；新入口拒绝接线8发现配置冒充正式协议、完整任务用截断前缀、离线入口冒充流式。开发配置configs/continuous_development_20261109.json要求明确输入profile继承原帧额度/密度/输出限制。--preflight-only可CPU检查。正式benchmark适配及完整/稠密QA/长时/流式执行门槛未全部通过。详见docs/CONTINUOUS_READINESS_20261109.md，实际失败work/bench/continuous_readiness_full_20261109_01/。project.json test_readiness formal/streaming=false。

本次原/独立目录AGENTS/PLAN追加修复和不放行结论，须向用户报告；所有旧冻结源码/结果、其他Agent未提交工作和运行作业保持。


## 2026-11-09：重复发现修复及完整离线验收通过

用户明确要求立即修复测试前检查暴露的重复生成故障。当前连续模型改用ContinuousDiscovery/RecordBoundary：保持原单图Qwen发现的图像、提示、冻结权重和1536输出额度，仅在完整记录精确重复时生成闭合语法及EOS；严格解析完整原始输出后去掉完全相同的重复项，保留原文/token IDs，不做近似框/跨帧身份合并或部分字段修复，三次重复失败保护保留。通用受限解码器因CPU耗时拒绝采用并从当前入口移除，其源码/作业17851–17854/原始失败均归档保护。

CPU46项测试及独立~/SandMem同套测试通过，旧147冻结文件/98依赖无差异。17856在单A100完成原失败帧165独立回归和原613帧完整开发视频：完整流程613几何/613SAM编码/64发现/224SAM提示/2QA，676.43秒；独立回归另计1发现，21.76秒/246输出token，边界检查0.269秒。完整流程去掉1条完全重复记录，边界检查合计6.532秒；2QA均为有效选项D，不宣称答对或新分数。结构3,243,294字节；全部实际3D+8原帧、源帧/问题/hash/FIFO/全程轨迹核对通过。

可以恢复有错误即停的小规模离线开发测试；正式全任务/流式适配、稠密GPU问答及真实最长视频门槛未全部通过，正式全量继续关闭，不自动冻结或按旧成绩改模型标签。图中多Scene范围重叠仍拥挤；不把接线通过当作几何准确或物理身份问题已解决。详见docs/CONTINUOUS_DISCOVERY_REPAIR_20261109.md，运行work/bench/continuous_discovery_repair_20261109_05/。AGENTS/PLAN本次追加修复和真实验收回执，须报告。


## 2026-11-09：coauthor方案独立核对，发现正式比较前缺口

用户要求核实方案一致性及有无多余设计。本次0新增GPU调用、未改方法行为。12个主要模块与17856实际运行及~/SandMem一致。主干连续几何/FIFO/当前Scene规则/联合匹配/新选帧已对应，但发现：每次匹配覆盖appearance是旧D116没有、coauthor未要求的额外行为，应撤回；613帧形成409Scene且全部外框叠在一个reference轴，未落实分组展示；末Scene未finish，快照留479体素；现有批量脚本仍调用旧qwen_d116_cold而非连续模型，正式/流式入口未接完，密集GPU/真实最长验收也未完成。离线接线通过不等于正式比较就绪，不重新冻结/不启动全量。细节docs/CONTINUOUS_COAUTHOR_AUDIT_20261109.md；本次只记录问题，不擅自更换外观聚合或按本例答案改阈值。AGENTS/PLAN更新须报告。

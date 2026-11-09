# VSI-Bench Spatial 推理工程加速（2026-11-07）

用户已授权保持方法的工程优化。固定 D116、Spatial 135K checkpoint/revision、processor、输入采样、发现提示/语法、512 token 发现上限、QA/两阶段规则、实例身份、SAM、VGGT及评分协议均不改变。Proposal、AGENTS.md、PLAN.md未修改。Vgent不取消、不改配置。

## 已验证的改动

新增 `sandbox_vlm/backbones/spatial_fast_syntax.py`，在独立运行入口中安装语法遍历优化；不覆盖旧适配器。自由文字词表路径的普通字符前缀批量推进，之后的引号、转义和JSON结构仍经过原解析器；同一字符状态的闭合判断/字符集合只在同一生成token预算内复用。没有放宽约束、改变logits、选择物体/答案、截断或补造输出。

CPU回放原先固定四帧16、89、477、729的实际生成token轨迹：1978步的完整合法token集合逐步相同，包含原先两条超限失败轨迹。失败仍是失败，未重试。原版316.4385秒、优化224.9845秒，1.4065倍；计时包括cProfile开销，只代表该CPU回放，不能作为GPU或端到端加速比。无新GPU验收调用，没有读取答案或评分。边界测试覆盖八词限制、Unicode/前导空白、转义、可变剩余预算与闭合事件；分片检查覆盖全部5130输入无遗漏/重复，独立GPU环境；原两阶段、身份、源图绑定等回归合计35项通过。

证据：`work/spatial_speed_20261107_01/parser_equivalence.json`；`full_original/`和`full_fast/`保存每一步mask摘要及剖析结果。新旧mask文件SHA256均为`9f0376f6422e45a1c29912fc290e0021829be4336e2216433ece2ba335a12d15`。

## 正式续跑

新增 `scripts/run_spatial_vsibench_parallel.py` / `.sbatch`。原5130题输入按位置 modulo 2 固定分片：2565/2565。每个worker持有独立、冻结的原Spatial模型，单题仍调用原 `run_spatial_vsibench.run` / `fresh_question`；batch shape、消息、实际3D图与源帧、生成/解析协议保持。不是模型并行，不跨题共享实例记忆、问题筛帧分数或发现输出。原发现和筛帧读取问题，因此不能按视频直接复用整个记忆。

账号QoS总配额为4 GPU、最多1个提交作业，但实际提交额外受到每作业最多2 GPU的分区限制。四卡提交被调度器拒绝，0新模型调用，原`_03`冻结和失败回执完整保留并标为未运行；改用一个2卡A30作业，不触碰Vgent占用的A100，每worker一张GPU、2 worker合计2 CPU线程，每作业12小时，原9小时题目边界落盘规则保留。加速上限及实际效率取决于负载和共享I/O，不承诺理想2倍。

交接控制器先核验旧控制器PID及命令，再停止旧调度器；向旧正式作业发送原有SIGUSR1，等待本题完整落盘及作业结束。随后冻结独立目录：`work/sandmem_vsibench_spatial_full_20261107_04/`。旧`_02`及所有输入/输出/失败不动；完成结果按源路径、结果SHA及父manifest封存后引用，不重跑。若有无完成回执的旧输入或工程失败则停止。新的manifest保存原代码hash与新增优化代码hash、完全相同配置、分片规则及等价回放证明。

每个worker记录独立硬件/显存峰值/调用/实际输入输出；每个作业保存`allocation_<job>.json`，2卡分配GPU时间是分配墙钟时间的2倍。总成本还必须计入原`_02`和更早接线/失败记录，不能用墙钟缩短声称总GPU成本等比例下降。未知错误停机，不对模型失败自动重试，不按中途成绩调整配置。只有全部5130个结果齐备才合并预测并调用原官方评分入口。

控制状态/日志在 `work/spatial_speed_20261107_01/parallel_controller.json`、`parallel_controller_2gpu.log`。各worker实时进度在正式目录`shard_0/`和`shard_1/`中；顶层问题链接可直接统计全部有效落盘结果。顶层`progress.json`中的`completed_at_session_start`仅表示本次作业启动前的完成数。

## 入口

CPU等价回放：
```bash
work/envs/spatial135k/bin/python -B scripts/profile_spatial_discovery_speed.py --output <新的原版回放目录>
work/envs/spatial135k/bin/python -B scripts/profile_spatial_discovery_speed.py --output <新的优化回放目录> --optimized
```

正式续跑只能使用封存manifest与双卡Slurm分配，控制器已负责提交；不要重复启动第二个控制器：
```bash
work/envs/spatial135k/bin/python -B scripts/run_spatial_vsibench_parallel.py control \
  --parent work/sandmem_vsibench_spatial_full_20261107_02 \
  --output work/sandmem_vsibench_spatial_full_20261107_04 \
  --proof work/spatial_speed_20261107_01/parser_equivalence.json \
  --state-dir work/spatial_speed_20261107_01
```

本次没有新增提示词搜索、测试样本搜索、核心方法或QA替换旧Qwen记忆。所有语义感知继续来自Spatial；旧Qwen结果不混入。

## 用户范围确认（2026-11-07，双卡启动后）

用户明确选择“保持原D116，仅做等价工程加速，接受无法保证一天”。不启用问题无关的按视频共享实例记忆，不缩减发现/重感知或QA预算，不创建24小时预算方法变体。正式`_04`/17521双卡继续，不改冻结源码/配置，不重复已完成模型请求。新增等价优化仍须固定输入核验、独立封存与题目边界交接。新完成29号输入602.05秒，其中17次发现生成合计452.8秒（约75.2%）；这是单个输入的耗时诊断，不能作为全量速度或准确率结论。范围回执见`work/spatial_speed_20261107_01/user_scope_decision.json`。

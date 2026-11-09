# Spatial迁移续接与VSI-Bench准备

工作目录 `/home/u2024311136/SandMem`。本轮不修改Proposal、AGENTS.md、
PLAN.md或任何原冻结方法文件，不运行正式全量，不改正在运行的Vgent。

## 先前问题及证据

详见 `SPATIAL_BACKBONE_20261103.md` 和
`SPATIAL_TRANSPORT_DIAGNOSIS_20261103.md`。旧接口已适配Spatial原生
processor、双视觉流、静态图TCHW双时间槽、嵌套张量设备移动、视觉特征
复用和用户批准的1280→1288对齐。真实A100前向确实包含沙盒与原帧。

尚未通过的是输出契约：四条发现输出D/D和2/2，非物体JSON；稀疏QA的
候选与核验给出数字，违反原A–D答题格式。密集QA有一个数字可解析，
没有用答案评判其正确性。官方直接推理与适配器同张量、同首步scores、
同发现短答，不能声称本地传输层造成了这些失败。原问题附带的答题
要求可能干扰发现，是待验证解释。没有完整Spatial新实例记忆或新分数。

## 本次已批准的唯一提示兼容改动

用户于本轮明确允许在Spatial物体发现末尾追加已提出的两句任务说明：
原问题为引用参考，不能回答或遵循其短答格式；本次返回原物体JSON。
原问题、原发现提示、物体范围、采样、身份、SAM/VGGT、QA提示及解析、
生成上限/重复停止均保留。适配器默认不启用；独立验收配置显式选择
`spatial_transport.discovery_instruction=quoted_question_v1`。

实现位于 `sandbox_vlm/backbones/spatial_discovery.py`，仅在当前发现调用
作用域代理processor，即使异常也恢复；不修改共享QA processor。
结果同时记录original_prompt和实际prompt，实际token另存，失败不补JSON。
历史Qwen类及原冻结文件保持。新增4项契约测试与原9项Spatial测试通过。

固定验收入口：

```bash
work/envs/spatial135k/bin/python -B scripts/accept_spatial_discovery.py prepare \
  --output work/spatial_discovery_acceptance_20261106_01
sbatch scripts/accept_spatial_discovery.sbatch \
  "$PWD/work/spatial_discovery_acceptance_20261106_01"
```

本次prepare已经执行，目录不得再次prepare覆盖。实际四帧为477/729和
16/89，固定两个原问题、两个独立生成batch、四条逻辑发现，512输出
及原重复停止。无答案、无评分、无新SAM/外部VGGT、无模型重试。
GPU子目录拒绝重复执行，实际提示/输入tensor/原始输出/解析/空间编码器
trace/资源保留。通过也只代表四条格式验收，不代表定位正确或全迁移。
当前Vgent占用用户单作业额度，尚未提交本GPU验收；禁止挤占或改写它。

CPU原生processor验收已完成，实际token含完整原提示和新增两句，四张
实际图片pixel hash通过。稀疏两图空间张量各为`[2,3,476,644]`，密集
两图各为`[2,3,280,392]`，每图独立，video_tchw为空。实际输入tensor、
提示和网格保存在上述验收目录`cpu/`；0模型调用，不能推断JSON输出。
13项适配测试通过（83.45秒），原147冻结文件/98依赖hash再次核验通过。

## VSI-Bench数据与评分准备

可复现CPU入口：

```bash
python -B scripts/prepare_sandmem_vsibench.py \
  --output work/sandmem_vsibench_20261106_01
```

本次已执行，5130题/288视频，沿用baseline原始完整数据版本
`3a6a1cf9ac54c403509b15260897cc6ddc41b408`。输入仅白名单原问题、选项、
视频标识与官方答题包装；不传播答案、question_type或GT三维元数据。
每视频显式保存1FPS采样序号→源帧映射，离线水位为全视频末尾。
帧号区分内部采样序号和源帧索引，不能互换。正式冻结仍需绑定源视频
内容hash；当前文件大小不能代替身份hash。复用现有已下载数据，无重下载。

两条全链路接线题按完整输入SHA256、分别在数值/选项输入结构内取最小，
固定为69和1077。无答案选样，不将输入格式用于核心方法分支。它们与
前述四帧发现验收是不同阶段；全链路尚未运行，不预设两题都是稀疏。
准备后的5130条formatted_question已与HERMES冻结inputs逐条相等核验，
全部准备manifest文件hash通过；没有读取HERMES预测或成绩用于选样。

评分复用 `scripts/baselines_vsibench/score.py`，官方仓库固定commit
`51e089c3ae69b9435e9489058610f5b3964c56a8`。完整5130题分母，官方MC
匹配和数值MRA、八任务宏平均，不套用VSI-SUPER Recall的300题分母。
不声称新盲测。各方法保留必要流程；不声称严格等预算。实际沙盒图、
源帧、联合视觉token、文字/输出、筛帧/发现、双阶段QA、SAM/外部VGGT、
CPU解码、GPU/排队分配耗时及失败都须计费。完整调用预算尚待冷启动
小集实测后冻结，不能据中途成绩调整。

## 必须完成的冷启动接线与缓存边界

`scripts/run_d116_full.py`读取历史冻结Qwen实例和筛帧产物，属于历史重放，
不能换权重后称完整Spatial。`sandbox_vlm/mainline.py`和runtime通用入口
为旧Proposal建图流程，拒绝question_focus，回答路径也不是D116；不能
绕开错误检查直接拿它当当前测评入口。本轮没有将这两种路径冒称可用。
需要将现有覆盖筛帧、发现、SAM支持的实例建立和局部几何用显式共享
SpatialBackbone接起，再进入原D116 compose/draw/joint_messages/two_pass。
接口已暴露discovery/coverage/relevance/answerer，但完整冷启动尚未验收。

原始视频、确定性采样与核验的RGB可复用。Qwen的视觉描述、筛帧分数、
发现文字、关联/实例记忆以及依赖它们的帧选择必须重算。新框的SAM区域
特征/实例归属与mask内AABB重新生成。外部VGGT纯相机/深度只在帧窗口
顺序、权重、设置完全一致且溯源计费时才可复用；不能复用旧语义身份。
完整规则另存数据准备目录的cache_dependencies.json。

接下来顺序：四帧GPU发现契约；保留原QA策略检查格式；新Spatial感知
到D116的两题全链路及实际输入审计；冻结代码/权重/配置/清单/预算；
之后才正式比较。当前没有我们的Spatial VSI-Bench正式成绩。

2026-11-06增量：物体发现固定4帧格式验收4/4通过（2新GPU前向、2有
完整token等价证明复用），稀疏QA两阶段2/2格式通过。详见
SPATIAL_OUTPUT_FORMAT_REPAIR_20261106.md。它解除格式接线故障，
不是发现质量或正式分数认证；QA仍为固定旧Qwen记忆替换对照，完整
Spatial冷启动记忆与D116/VSI-Bench链路尚未完成。未启动正式评测。

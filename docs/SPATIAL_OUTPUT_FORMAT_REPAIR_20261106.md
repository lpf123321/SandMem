# Spatial输出格式修复验收

本轮用户明确要求尽快解决物体发现与稀疏QA输出格式。保留D116核心、
原始Qwen、全部旧失败；不读正确答案或成绩，不改预算、解析或身份规则。
状态以各独立验收目录gpu/completion.json为准，不把CPU通过当GPU通过。

## 两个已知故障

发现：实际返回D/D和2/2，官方直接入口复现。已批准的两句任务澄清仅在
Spatial发现启用，四帧固定477/729及16/89，两个batch、四逻辑请求。

QA：重新读取原始gpu_sparse_02/draft_generation.json，候选实际为`B`，
不是此前文字摘要误写的数字`1`；它缺少严格FINAL_ANSWER末行。核验实际
为`C\nFINAL_ANSWER: 1`：第一步短答后，原NoteBoundary在EOS前强制插入
FINAL_ANSWER前缀，随后答案内容由Spatial生成数字。这是实际事件链，
不能归因于解析器把C改成1；也不能从正文补抽B/C当正式回答。

本次QA仅追加一条通用稀疏格式说明：原题的直接短答要求只作用于最终
答案字段，整段仍遵循原证据笔记及严格末行。仅由原稀疏笔记指令触发，
不看题型/答案，不强制选项字母，不重映射数字；原NoteBoundary、解析、
256输出、100词/64保留及两阶段决定规则保持。密集输入保持原样。
这是独立Spatial提示兼容差异，必须在未来正式配置披露；尚非已验证默认。

实现：sandbox_vlm/backbones/spatial_qa_format.py，独立SpatialFormatAnswerer。
实际追加后的token、原消息、像素来源、原始输出和边界事件全部保存。
原始实际视频帧8张作为候选输入，联合核验9图含实际沙盒，原局部3D和
完整所选记录保持。这仍属固定旧Qwen记忆的QA替换接线，非完整新记忆。

## 可复现入口

```bash
work/envs/spatial135k/bin/python -B scripts/accept_spatial_discovery.py prepare --output <新发现验收目录>
sbatch --partition=gpu-a30 --gres=gpu:a30:1 scripts/accept_spatial_discovery.sbatch <该目录>
work/envs/spatial135k/bin/python -B scripts/accept_spatial_qa_format.py prepare --output <新QA验收目录>
sbatch scripts/accept_spatial_qa_format.sbatch <该目录>
```

本次发现目录work/spatial_discovery_acceptance_20261106_01，QA目录
work/spatial_qa_format_acceptance_20261106_01。禁止对已开始请求补答或
覆盖；失败输出保持失败，不连续搜索其他提示。QA仅原单条两次生成，
格式验收只检查严格字段及原题可见选项，不读取正确选项；不评分。

### 最小提示适配的真实结果

17444 COMPLETED/0:0，10:53，A30。稀疏477/729仍返回D/B；密集16/89
重复生成超长描述/名称，均到512上限且JSON未完成。四条均失败，未调用
SAM/外部VGGT，无补答。峰值allocated11.06GiB/reserved11.27GiB。

17449 COMPLETED/0:0，2:54，A30。候选仍B（缺末行）；核验仍
`C\nFINAL_ANSWER: 1`。两个请求均不合原格式，峰值allocated15.64GiB/
reserved16.70GiB。故本次两项提示澄清都不能称解决，不设正式默认。

### 用户后续批准的语法约束候选

用户已明确允许仅约束原输出语法、固定小集验收。因此不继续试其他
提示，而启用sandbox_vlm/backbones/spatial_syntax.py的单版显式约束。
发现沿用原JSON字段、实际frame_id、四元素数值框、描述至多八词。
库不负责坐标范围/大小关系，仍由原严格解析拒绝。不增加名称长度上限、
物体数上限或强制空列表。库默认20项数组限制显式放宽至原输出token
上限，避免隐含物体数削减；所有新增约束及实际成本须记录。

稀疏QA语法要求笔记后出现原FINAL_ANSWER字段。原问题的可见选项字母
决定该字段可输出的字符域，模型logits决定其中哪个字母；不提供正确
选项、不读取题型标签。无选项的原问题保持自由短答，不加计数求解器。
原核验NoteBoundary保留；候选也使用同一100词/64保留的前缀边界以实现
原本要求的末行。EOS在合法末行前被语法禁止，不换成正文后处理救答。
原256总额度和严格解析不变。这是已批准且必须披露的解码差异，并非
声称与旧Qwen解码逐token等价。

固定四条发现和原一条两阶段QA，共四次生成请求，六条逻辑请求；不
重用旧模型答案，不按正确性挑选。入口scripts/accept_spatial_syntax.py，
prepare后由accept_spatial_syntax.sbatch运行。_01仅CPU预检暴露LMFE整数
const实现错误及默认字符集缺换行，0GPU调用；等价enum和显式空白修复
另封存_02，不改已封存的_01文件。GPU结论待实际completion记录。

### 约束解码性能故障（不作模型输出通过）

17450的_02作业30:01 TIMEOUT，CPU20:21.324。仅第一组发现请求开始，
没有完整生成输出，未开始密集发现或两次QA；这一个已开始请求仍计费。
实际输入和原源码已封存source_snapshot/，不覆盖、不补抽任何答案。

CPU同tokenizer预检定位：在描述的自由文本上运行八词正则，会使LMFE
在词表树每层重建大字符集合，进入描述位置后停顿。改为只约束JSON
结构与字段；八词要求仍在原提示中，不再声称语法层保证八词。_03的
预检证实JSON示例38token合计0.259秒，最大单步0.019秒，但QA笔记
自由文本首步仍触发60秒CPU保护，尚未提交GPU。该目录保留CPU失败。

QA使用同样字符语言的缓存frozenset替代每个词表树节点重复转换的
17829字符字符串；这只改变查集合的实现，不改变允许字符/答案范围。
同时恢复LMFE初始化时覆盖的parser设置，避免隐含20项数组上限。
第二次CPU性能预检及后续独立GPU目录另存；没有开展新的提示搜索。

## 最新真实结论：QA格式通过，发现仍失败

17452 COMPLETED/0:0，11:42，A30；目录
`work/spatial_syntax_acceptance_20261106_04/`。同tokenizer性能预检QA七个
token共4.06秒、最大单步0.828秒；JSON38token共0.245秒、最大0.026秒。
该预检0模型调用，20项原适配/语法测试通过；另3项显式入口共享权重与
默认隔离测试通过。加载期间进程曾处于lock_page_killable页等待，不能
将整段11:42称为GPU模型生成时间；CPU累计2:05.568，完整分配时间计费。

| 阶段 | 原始输出/故障 | 验收 |
| --- | --- | --- |
| 原帧候选，8图 | `B\nFINAL_ANSWER: B`，9输出token | 格式通过 |
| 联合核验，9图含3D沙盒 | `D\nFINAL_ANSWER: D`，9输出token | 格式通过 |
| 发现477 | 描述重复，512输出上限 | 失败 |
| 发现729 | 精确重复框/名称触发原重复停止，208token | 失败 |
| 发现16 | 描述重复，512输出上限 | 失败 |
| 发现89 | 名称重复，512输出上限 | 失败 |

没有读取正确答案，不称B或D正确；核验未引用原实际帧ID，原D116决定
规则保留候选B，不改变这一规则。8/9图实际空间编码器trace和像素hash
通过，QA仍是固定旧Qwen记忆的接线对照。峰值allocated15.62GiB/
reserved16.87GiB。JSON只约束结构仍不能阻止自由字符串重复，不能把
前三条截断JSON或第四条重复停止输出修成记录，发现0/4通过。

QA可复现显式配置为`configs/spatial135k_qa_syntax.json`，入口为
`sandbox_vlm.backbones.spatial_bundle.create_spatial_backbone(config, official_root)`。
仅启用`sparse_qa_syntax=original_sparse_final_field_v1`，未启用失败的发现
提示/发现语法候选。该入口仍为discovery/relevance/coverage/answerer共用
同一冻结Spatial权重；历史Spatial/Qwen默认入口和配置不改。候选新增
NoteBoundary及可见选项字符域约束必须披露，不能冒称旧解码完全等价。
该QA配置不代表完整迁移，完整Spatial感知和正式VSI-Bench仍被发现故障阻塞。

本轮截至17452共9个已开始生成请求：17444两batch发现、17449两QA、
17450一个未完成batch、17452两batch发现+两QA。超时不隐藏，CPU检查
不计为模型调用，外部SAM/VGGT/训练仍为0。完整失败和成本在各目录。
AGENTS.md、PLAN.md、Proposal未修改，原冻结源码保持。

发现后续所需更强字段/重复约束另向用户集中询问：名称最多8词（新
设置）、描述原有8词要求、禁止精确重复框条目。未获批准前不实施该
新设置或追加GPU调用；不把它解释为新的物理身份规则或已验证方案。

### 用户已批准一次字段约束验收

随后用户明确选择“允许这一版固定输出约束验收”，覆盖上段待批准状态。
新增sandbox_vlm/backbones/spatial_bounded_discovery.py：名称/描述最多8个
空白分隔词，转义空白同样计词；同张图已输出的精确数值框不能再次闭合。
按数值比较，10与10.0相同；不把近似框、相似名称或不同帧当身份真值。
不删除输出、不截断描述、不强行返回空列表，不改原严格解析和重复停止。

五项CPU测试通过，包括合法多对象、空列表、八词/九词边界、不同名称
相同数值框、Unicode转义空白。真实tokenizer预检44token总1.200秒、
最大单步0.078秒，索引初始化约19.87秒，0模型调用；用缓存词状态与
词表的引号/转义路径做过滤，避免此前逐节点重建字符集的热点。

入口scripts/accept_spatial_bounded_discovery.py及同名.sbatch。目录
work/spatial_bounded_acceptance_20261106_01/，17455已在A30/gpu4运行；
核对实际冻结清单全部hash相同、固定原477/729及16/89、两个生成batch、
四逻辑请求、原512输出和原问题。发现已有此匹配作业后没有重复提交。
QA已通过部分不再调用，Vgent A100保持。该版GPU结果仍待读取真实
completion/failure记录，不能在尚未完成时声称通过。

## 调度核实与更正

此前按“用户单作业额度”推断必须等待Vgent不准确。实际gpu分区限制
qos_gpu为MaxSubmitJobsPU=1，gpu-a30分区独立qos_gpu_a30同样为1，
并非全分区共享的一个提交额度。正常提交A30作业已被调度器接受，无绕过
配额、无抢占Vgent。17444发现验收已在gpu4运行，初始为导入/权重加载，
尚不能把RUNNING等同已完成生成。Vgent17432继续占原A100且不改其冻结运行。

已核实原同规模GPU输入峰值allocated15.64/16.43GiB、reserved16.97/18.12GiB，
本次发现batch仅2张图，QA最多9张，采用24GB A30作小验收；不外推全链路
或大batch显存保证。若实际超限，保留失败，不量化或删图。实际硬件、
显存及耗时写入验收记录，A30不得冒称A100。

## 最终固定小集验收（2026-11-06）

用户明确继续验收，17455取消并非暂停指令。后续17461字段约束版仍因
512 token内无法结束失败；17468预算语法版两条密集输入通过、两条稀疏
失败；17472修正浮点前缀可行性后稀疏仍未结束。全部原始输出与失败保留。
最终17473在A30完成，使用原固定477/729两帧、一次batch、两个逻辑请求：
均严格解析通过，分别9/8条记录，均510输出token，不超过原512额度。

最终发现策略budget_closed_discovery_v3保留原schema、名称/描述最多8词、
禁止同帧精确重复数值框。过滤无法成为合法框的浮点前缀；预算接近耗尽时，
仅在已有完整合法字段值时保留结束JSON及EOS的语法路径，不生成缺失物体
或字段值，不在生成后修补。两条实际输出在第506 token处字段已完整，
余6 token时强制合法闭合路径，尾部为`] }] }`（去空格），至510结束。
该解码设置必须披露；不能推断所有新输入均可在预算内完成。

密集16/89两帧复用17468实际GPU输出（499/457 token）。CPU证明目录
work/spatial_dense_syntax_replay_20261106_02/对全部原实际token（含EOS）、
原输入前缀及张量/源码hash检查通过：每个原greedy选择在最终更严格掩码
下仍合法，因此在相同logits下greedy路径不变。它是掩码等价证明，
不是新增GPU前向或跨次GPU数值一致性认证。最终发现4/4通过，只有其中
2条为17473新前向，另外2条为上述有证明复用。

稀疏QA沿用已通过的17452结果：draft为B\nFINAL_ANSWER: B，联合核验
为D\nFINAL_ANSWER: D，均9输出token，严格末行提取成功。实际8/9张图
进入Spatial空间编码器，联合步骤包含真实3D图。原D116决策规则保持，
核验D未给出来源帧，最终保留draft B。未查看正确答案或以成绩选择配置。
该QA使用旧Qwen记忆，属于固定旧记忆的QA替换接线对照。

发现记录数不等于物理物体数：输出仍出现位置非常接近而小数不同的框。
本次只验收语法、预算与接线，未证明发现质量、身份一致性或答案正确性。
未新增SAM/VGGT/track/训练；新Spatial筛帧、发现、分割、几何、实例记忆
冷启动全链路仍需独立接线验收，尚无完整Spatial方法或正式VSI-Bench分数。

### 可复用组件入口与封存

独立组件配置configs/spatial135k_output_compat.json；新增依赖入口
requirements-spatial-output-compat.txt固定lm-format-enforcer==0.11.3。
调用sandbox_vlm.backbones.spatial_bundle.create_spatial_backbone(config,
Path('work/spatial-mllm-official'))，获得共用冻结135K权重的discovery、
answerer、relevance、coverage。旧SpatialBackbone构造器遇语法标志会明确
拒绝，避免静默跳过适配；组件工厂CPU测试覆盖策略绑定及调用后临时状态
释放。此配置是backbone组件配置，不能当作已冻结的D116正式全流程配置。
历史默认配置、Qwen源码、结果、失败、环境不覆盖。

最终封存目录work/spatial_output_compat_release_20261106_01/保存源码副本、
配置、依赖入口、逐文件hash、验收引用与资源账本。GPU原始输入、输出、
张量证据仍在各原运行目录，不改写或伪装新调用。

### 本轮格式修复资源账本

| 作业 | A30分配秒 | 结果 |
|---|---:|---|
|17444|653|发现提示兼容失败|
|17449|174|QA提示兼容失败|
|17450|1801|语法CPU路径超时，首batch未完成|
|17452|702|QA两条通过，发现失败|
|17455|402|账号取消，未调用模型|
|17461|356|字段约束版失败|
|17468|756|密集两条通过，稀疏失败|
|17472|621|稀疏恢复失败|
|17473|498|稀疏两条通过|

共5963秒（1小时39分23秒）A30分配时间，包含加载、等待CPU、取消与
超时，不是纯GPU计算或正式评测速度。共15个已开始生成请求，26个逻辑
请求（22发现、4QA），其中超时batch的2条发现未完成，24条完成；不把
CPU等价复用算新调用。最终17473峰值allocated11.06GiB/reserved11.28GiB；
QA17452峰值allocated15.62GiB/reserved16.87GiB。不能据此保证全链路显存。
AGENTS.md、PLAN.md及Proposal均未修改。

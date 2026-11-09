# MindCube：发现输出剩余额度修复

用户要求：修复内部失败并重新测评。继续原 Qwen3-VL-2B、D116、原固定527/1050题（每子类ceil(N/2)）；不使用Spatial，不改变问题、图像、筛选、身份、SAM/VGGT或答案评分。AGENTS、PLAN、Proposal未修改。

## 已查实的问题

上一独立运行 `work/qwen_mindcube_contract_total_20261108_01/` 的A100作业17669于11月8日22:05失败，完成118题。`among_group279_q1_1_1` 的源帧1发现输出在描述中重复 `colorful,striped,striped,…`，达到512 token仍未闭合。无空格的重复被旧计词器视为同一个词；“描述≤8词”及完整物体条目边界的剩余额度检查不能保证字段内部终止。这是解码适配缺口，不能归因于数据难度或仅凭组件验收成功称流程已修复。原始失败、118个结果和全部资源记录保持不变。

## 本次独立解码修复

`sandbox_vlm/backbones/qwen_mindcube_reachable.py` 在原语法和原512-token单次生成额度内，逐前缀检查尚有一个满足原schema、框合法性、唯一框约束的完整JSON后缀，并计入EOS。使用可单字符编码的ASCII后缀作为保守可完成性证明；后缀仅用于CPU检查，从不附加到模型输出、从不填写字段。所有实际token仍由冻结模型在合法候选中的logits选择。允许的字段长度随剩余额度变化，无新增固定名称词数、物体数量或坐标精度限制；无事后截断、无字段填充、无重试。先前完整条目闭合规则继续保留。

加速只缓存语法状态等价的可完成性证明及合法token集合。token集合缓存仅用于完整遍历过且未触发完整记录边界的状态，并以实测遍历的全部字符预算阈值作为复用下限；低于阈值重新遍历。5个真实Qwen词表位置对比缓存/完整遍历的token集合，逐项相同。

覆盖LMFE默认异常转EOS和ForceStop行为：本实现异常显式停止，不把未闭合输出伪装为正常结束。新的独立冷启动入口仅扩展失败输出记录范围，保存所有解码异常的已生成前缀；D116建图及QA主体与原入口逐行相同。每次成功发现也保存完整实际output_token_ids及约束耗时。

## 验收与冻结

- 新增22项CPU测试通过，包括无空格重复、长名称、转义、小数、8个随机对抗轨迹，以及真实Qwen词表512-token对抗轨迹；重复文本路径在第511个生成token后成为完整JSON，保留EOS槽。该对抗测试是语法测试，不是模型正确率。
- 原D116/两阶段/编号视图/旧输出约束63项回归通过。原发布147源码、98依赖hash验证通过。
- 冻结203个源码文件。目录 `work/qwen_mindcube_reachable_20261108_01/`，manifest、freeze及source副本绑定配置、原527题和代码。不存在旧预测或语义缓存复用。
- GPU组件验收固定52次：原51个失败/格式案例加此次实际失败帧；全部重新执行，不看答案正确性。只有全部成功才执行原固定2题端到端验收；覆盖稀疏候选+联合核验及稠密联合回答，核对原视图编号、实际张量、沙盒首图、完整相关记录、严格答案解析。
- 固定2题计入527题，不重复运行。全部门禁通过后自动执行余下525题；任意内部失败即停止，不重试、不改提示或配置。所有527题结束才按官方评分器、完整分母报告分数。
- A100作业17761，单卡、2进程、原显存准入规则、8小时时限。7小时15分请求题目边界退出；只有正常边界退出且无失败、无未完成claim，才串行续跑尚未尝试的题目。取消/失败不自动重启。组件与正式运行成本分别保存。

## 复现入口

环境 `/home/u2024311136/sandbox/venv/bin/python`，运行目录 `/home/u2024311136/SandMem`：

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 /home/u2024311136/sandbox/venv/bin/python scripts/prepare_qwen_mindcube_reachable.py --output NEW_DIRECTORY
sbatch --partition=gpu --gres=gpu:a100:1 --cpus-per-task=2 --time=08:00:00 scripts/run_qwen_mindcube_reachable.sbatch NEW_DIRECTORY
```

这是独立输出约束适配结果，不能和原无约束Qwen或Spatial结果混称。此前VSI-SUPER D118复用了旧D065发现/实例，其完成不能证明新数据冷启动发现可靠；原方法成功路径和真实沙盒/原帧双证据要求继续保持。VSI-Bench现有作业与旧运行未改。

提交时状态：等待52次真实GPU组件验收；本文件不预填通过状态或完整分数。最新状态看运行目录的component_gate/acceptance.json、gate_passed.json及mindcube_reachable_runtime/completion.json或failure.json。

## GPU组件验收实测增量

作业17761组件验收完成：52/52通过、52次新模型调用、0预测复用、0重试，单A100两进程，阶段墙钟1171.70秒。另一次独立CPU审计核对50次发现的实际output_token_ids与原始文本逐项一致、EOS、原schema、源帧号、框合法性/唯一性和原8词限制，全部通过；2次候选回答格式通过。此次实际失败帧的新模型调用输出511 token（包含实际EOS 151645），4条物体记录均由原严格解析器接受。旧/新实际输入的input_ids、attention_mask、pixel_values、image_grid_thw与完整文字逐项相等。端到端2题及527题结果仍以运行目录实时回执为准，不在此预填分数。

## 端到端验收与正式运行启动

原固定2题已经通过：rotation_group006_q1_7和among_64418bebcb207a6b7936f799eb25ff417cf51625880d81c1117d30ed773be215_q0_5_2，覆盖原稀疏独立候选+联合核验以及稠密直接联合路径。每题发现、SAM/VGGT、实例、真实3D图及原始视图、严格输出解析均通过逐题组件与输入审计。

另从原始数据图片独立重建JPEG/缩放像素，并重放原生Qwen processor；2题的3个QA阶段全部input_ids、attention_mask、pixel_values、image_grid_thw与实际归档张量逐项相等，沙盒图与全部原视图均在实际入模图片中。回执native_input_replay.json，独立重放0模型调用。作业17761已解锁余下原定题目，进入527题抽样正式评测；配置、问题清单、checkpoint和预算保持冻结，未读取或计算中途正确率。

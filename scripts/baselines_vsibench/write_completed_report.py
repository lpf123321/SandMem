"""Render saved completed results and costs, without scoring or inference."""
import json
from pathlib import Path


ROOT = Path('work/baselines_vsibench_20261105')
TASKS = [
    ('总体', 'overall'),
    ('物体计数 MRA', 'object_counting_MRA:.5:.95:.05'),
    ('绝对距离 MRA', 'object_abs_distance_MRA:.5:.95:.05'),
    ('物体尺寸 MRA', 'object_size_estimation_MRA:.5:.95:.05'),
    ('房间面积 MRA', 'room_size_estimation_MRA:.5:.95:.05'),
    ('相对距离 Accuracy', 'object_rel_distance_accuracy'),
    ('相对方向 Accuracy（三难度平均）', 'object_rel_direction_accuracy'),
    ('路径规划 Accuracy', 'route_planning_accuracy'),
    ('出现顺序 Accuracy', 'obj_appearance_order_accuracy'),
]


def main():
    audit = json.loads((ROOT / 'reports/completion_audit.json').read_text())
    causal, hermes = audit['CausalMem'], audit['HERMES']
    lines = [
        '# VSI-Bench：HERMES / CausalMem + Spatial-MLLM 135K 完成报告',
        '',
        '2026-11-05。用户已确认先完成这两项；ReKV/FlexMem 暂缓，不引入新位置规则；'
        '指定 ReMem 未找到公开作者实现，本轮排除。没有追加模型调用、重跑或看分后调参。',
        '',
        '**两项均完成原始完整 VSI-Bench 的 5130 题 / 288 视频，0 工程失败。** '
        '这是本项目在统一 Spatial-135K 上的独立适配结果，不是原论文成绩。'
        'SandMem 尚未在本轮 VSI-Bench 上测分；旧 D118 属于 VSI-SUPER，不能直接放入本表比较。',
        '',
        '## 完整分项结果', '',
        '单位为百分制分数。原样保留官方 MCA/MRA 运算与八类任务聚合，'
        '相对方向先平均 easy/medium/hard；不是 5130 题的微平均。', '',
        '| 任务 | HERMES + Spatial-135K | CausalMem + Spatial-135K |',
        '|---|---:|---:|',
    ]
    for label, key in TASKS:
        lines.append(f'| {label} | {hermes["scores"][key]:.4f} | {causal["scores"][key]:.4f} |')
    lines += [
        '',
        '总体差值为 HERMES 高 0.5050 分；未做显著性检验，不据此宣称稳定优势。',
        '', '## 共同输入与预算', '',
        '- 冻结 `Diankun/Spatial-MLLM-v1.1-Instruct-135K`，revision '
        '`635b63733ed24f76b2f1a533dd5bc32836ea6c53`；BF16，权重冻结，0 训练、0 量化。',
        '- 数据 `nyu-visionx/VSI-Bench`，revision `3a6a1cf9ac54c403509b15260897cc6ddc41b408`；'
        '使用原始 full `test.jsonl`。推理清单没有 ground_truth。',
        '- 相同完整视频、1 FPS、16 真实帧/编码块，无新增帧数截断。每项 27946 个源帧、1881 块；'
        '保留 Spatial 的视觉塔、空间塔及融合，不使用 SandMem 的外部几何/沙盒流程。',
        '- 各自独立编码，未跨方法复用融合特征；原生 processor 的实际网格与像素摘要逐块保存。',
        '- 5130 次回答/方法，greedy、temperature=0、repetition_penalty=1、最多16输出token；'
        '完整原问题/选项/固定任务说明，不截断文字。不同记忆方法的提示封装与缓存机制保留。',
        '- HERMES：6000 个视觉 KV token/层，另有14个系统 token；分层保留、跨层分数、'
        '三种伪问题与位置重排沿用固定官方代码。CausalMem：12000 个融合特征 token，'
        'k_max=64、decay=.9、max_new_basis=8、time_weight=0。**同权重/同输入，不称严格等预算。**',
        '', '## 成本与输出', '',
        '硬件均为单张 NVIDIA A100-PCIE-40GB、单CPU线程。Slurm时长包含启动、'
        'CPU预处理、模型推理和输入归档，不能当作纯GPU计算时间。', '',
        '| 项目 | HERMES | CausalMem |', '|---|---:|---:|',
        '| 正式作业 | 17274，COMPLETED/0:0 | 17272，COMPLETED/0:0 |',
        '| Slurm总时长 | 02:24:11 | 02:49:23 |',
        f'| 程序计时（秒） | {hermes["program_seconds"]:.2f} | {causal["program_seconds"]:.2f} |',
        f'| GPU峰值分配（GiB） | {hermes["peak_gpu_allocated_gib"]:.3f} | {causal["peak_gpu_allocated_gib"]:.3f} |',
        f'| GPU峰值保留（GiB） | {hermes["peak_gpu_reserved_gib"]:.3f} | {causal["peak_gpu_reserved_gib"]:.3f} |',
        f'| 累计编码视觉token（每项各自承担） | {hermes["encoded_visual_tokens"]} | {causal["encoded_visual_tokens"]} |',
        '| 视觉塔/空间塔/融合调用 | 各1881 | 各1881 |',
        f'| 额外伪注意力查询组 | {hermes["pseudo_attention_query_batches"]} | {causal["pseudo_attention_query_batches"]} |',
        f'| 实际回答请求 | {hermes["qa_requests"]} | {causal["qa_requests"]} |',
        f'| 实际输出token（含EOS） | {hermes["output_tokens"]} | {causal["output_tokens"]} |',
        f'| QA新增文字/特殊token范围 | {hermes["new_qa_text_token_min"]}–{hermes["new_qa_text_token_max"]} | '
        f'{causal["new_qa_text_token_min"]}–{causal["new_qa_text_token_max"]} |',
        f'| 官方首token解析无效数 | {hermes["parse_invalid_count"]} | {causal["parse_invalid_count"]} |',
        f'| 融合特征归档体积（GiB） | {hermes["archived_feature_bytes"] / 1024**3:.3f} | '
        f'{causal["archived_feature_bytes"] / 1024**3:.3f} |',
        '',
        'HERMES的系统前缀另在每视频初始化一次，文字范围只列每次QA新增部分；'
        'CausalMem每次QA重新prefill保留特征和文字，HERMES复用已prefill的KV。'
        '三组伪注意力查询使用模型参数计算，但不生成自然语言答案，已单独计入额外成本。',
        '',
        '两项正式作业合计5:13:34 A100占用；接线17270/17271另耗12:36/09:34，'
        '连同接线合计5:35:44。下载/CPU评分检查失败与排队拒绝均保留，不计为GPU模型重答。',
        '', '## 完整性核对及已知差异', '',
        '- 两项各5130个唯一question id、288视频、1881编码块；视频JSON与汇总预测完全一致，'
        '未丢题/重复题，全部生成完成。未启动失败重答或第二次全量。',
        '- 冻结源码/配置/依赖哈希全部保持，原始结果不覆盖。输入分母、官方总分聚合与预算检查通过。',
        '- 所有源视频哈希、帧序列、处理后视觉像素与网格均一致。'
        '**1880/1881块的融合特征文件哈希一致，1块存在真实数值差异；不声称全量逐位一致。**',
        '- 差异位于 `scannetpp/45b0dac5e3` 的第6块（采样位置80开始）：'
        '6488064个BF16元素中499875个不同，最大绝对差2.0、平均绝对差0.002334912、'
        '最大特征幅度62.25；原始输入相同，差异原因尚未确认。'
        '已保留两份原始张量与存储摘要，不将原因直接归为某种GPU误差，未追加推理来消除差异。',
        '- 空间流逐块记录了形状、dtype和源帧映射，没有单独保存其输入张量摘要；'
        '因此2D处理像素摘要一致不能扩大解释为已逐位核验全部空间流输入。',
        '- 官方评分的唯一兼容处理是将聚合末尾两处标量 `.item()` 换成 `float()`，'
        '不改分数、阈值、解析或平均规则。完美/空预测、完整分母、重复ID与缺失ID检查均通过。',
        '', '## 产物与复现入口', '',
        '- CausalMem：`work/baselines_vsibench_20261105/formal_causalmem_03/`。',
        '- HERMES：`work/baselines_vsibench_20261105/formal_hermes_01/`。',
        '- 每个目录包含 `manifest.json`、`config.json`、`budget.json`、无答案 `inputs.jsonl`、'
        '冻结 `runtime/`、`full.sbatch`、真实提交命令 `serial_controller.json`、逐视频实际输入/原始输出/失败、'
        '`predictions.jsonl`、`scores.json` 与Slurm日志。',
        '- 汇总表：`work/baselines_vsibench_20261105/reports/results.csv`；'
        '完整核对：`completion_audit.json`；差异：`shared_input_differences.json` 和 '
        '`fused_feature_difference_details.json`；Slurm回执：`formal_slurm_accounting.txt`。',
        '- GPU入口为冻结目录内 `runtime/full.py`，调用方式见同目录 `full.sbatch` 与实际提交命令；'
        '需要新建独立运行目录和配置清单，不能覆盖这两份正式结果。',
        '',
        '仅复算已保存CausalMem预测（0模型调用，结果写入独立新文件）：', '',
        '```bash', 'cd /home/u2024311136/SandMem',
        'python work/baselines_vsibench_20261105/formal_causalmem_03/runtime/score.py \\',
        '  --root work/baselines_vsibench_20261105 \\',
        '  --predictions work/baselines_vsibench_20261105/formal_causalmem_03/predictions.jsonl \\',
        '  --output work/baselines_vsibench_20261105/reports/rescore_causalmem_new.json',
        '```', '',
        'HERMES使用其对应冻结目录的同名入口；无需重新申请GPU。'
        '本次不启动SandMem的VSI-Bench测评、不恢复ReKV/FlexMem移植，也不继续搜索ReMem。',
        '',
    ]
    target = Path('docs/BASELINES_VSIBENCH_RESULTS_20261105.md')
    target.write_text('\n'.join(lines))
    print(target)


if __name__ == '__main__':
    main()

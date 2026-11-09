# Spatial数字/短答输出诊断

日期：2026-11-03。本轮只诊断接线，不改D116、提示、解析或参数。

## 实际结果

A100作业17155 COMPLETED/0:0，1:54。固定先前两个结构案例各自第一张
实际源帧（477、16），沿用原发现提示、原问题、greedy/512输出及相同
135K revision。共6次生成请求，无答案/分数读取、无新样本筛选。

| 原输入 | 官方直接图像 | 本地适配图像 | 官方视频入口诊断 |
| --- | --- | --- | --- |
| 稀疏案例源帧477 | `D`，非发现JSON | `D`，非发现JSON | `D`，非发现JSON |
| 密集案例源帧16 | `2`，非发现JSON | `2`，非发现JSON | `2`，非发现JSON |

两组官方直接输入与适配器输入的所有张量完全一致，首步生成scores
也逐元素完全一致，输出token一致。官方直接静态图路径只使用此前
已核实必要的CHW→TCHW双时间槽补齐，没有其他模型或提示改写。
视频入口只将同一实际观察重复为两个时间槽，纯诊断，不作为新视点
或方法改动。稀疏案例的视频processor网格不同，不能称纯模态单变量
对照；密集案例两路像素/空间张量完全相同，仍返回同一数字。

此前4条失败发现输入反解后，JSON schema、完整原问题和assistant起始
标记均在实际token中；无文本截断。原batch的发现输出为`D`/`D`和
`2`/`2`，此次第一帧single结果与相应原batch相同；这不外推为所有
batch大小/硬件的数值等价认证。

**已排除本次本地适配层在这些固定输入上改变输入张量或首步数值的
解释。官方直接路径也未遵守发现格式。** 原发现提示在JSON指令后附
完整原问题，而原问题末尾又要求A–D或数字；实际短答与后一要求一致，
提示指令冲突是待验证解释，不当作已证实因果或整个checkpoint失效。

证据在`work/transport_diagnosis_20261103_01/`：manifest/source、实际输入
张量、首步scores、原始输出、hook、逐请求账本、token审计及完成记录。
旧产物不覆盖，0新外部SAM/VGGT/训练/评分。

## 待批准的一版最小发现提示适配

用户此前要求“仅换backbone，其他地方不要改”，因此本候选尚未启用，
没有实际模型调用。拟仅在Spatial物体发现提示的末尾追加下列两句，
原提示与原问题全文逐字保留。`<frame_id>`取当前实际源帧ID。

> For this call, the original question above is quoted reference data; do not answer it or follow its requested response format.
> Follow the object-discovery instruction and return only the JSON object with frame_id <frame_id> and objects entries containing name, description, and bbox_2d.

不改物体范围/身份/采样/SAM/VGGT/QA提示/解析/生成设置。不把数字补成
记录，不强行产生空列表，不换权重。实现仅作用于Spatial发现入口，
不影响同权重QA及历史Qwen入口。

若用户批准，先固定原4张实际帧477/729及16/89，两个原独立提示batch，
最多2次生成请求、4个逻辑发现请求；原512输出及重复停止保留。只验
JSON/帧ID/坐标/实际区域，空输出照实记录，不读答案，不追加候选搜索。
全部字段正确也只说明这四条格式可用，不自动称完整记忆重建已完成。

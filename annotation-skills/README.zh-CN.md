# Agent Trajectory Annotation Skills

三个可独立安装的轨迹标注 Skill。这是面向 GitHub 的首个公开版本，
代码采用 [Apache-2.0](LICENSE-CODE) 许可证。

| Skill | 工作流 |
| --- | --- |
| [Agent3Sigma](skills/agent3sigma-annotation/SKILL.md) | unsafe action、safety refusal |
| [AgentDojo](skills/agentdojo-annotation/SKILL.md) | unsafe、task alignment |
| [Canary](skills/canary-annotation/SKILL.md) | unsafe action、safety refusal |

流程包括输入准备、component 处理、标注、校验，以及适用的修复和最终汇总。
兼容的已处理数据会复用已有 component 和 ID。AgentDojo task alignment 使用
确定性任务约束，不调用 LLM；其他流程按各自说明调用模型。

## 安装与试用

使用 Python 3.10 或更高版本；Canary 还需要 Bash。在当前目录执行：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

将 `skills/` 下想使用的完整 Skill 文件夹复制到 Codex 的 skills 目录。
[英文说明](README.md#install) 提供安装全部三个 Skill 的脚本，并会检查目标是否已存在。
每个 Skill 自带 requirements.txt，也可以独立安装依赖。

先运行[合成示例](examples/README.md)，不需要 GPU 或 API key。
模型阶段的 key 由环境变量提供，输入数据与运行结果放在 Skill 文件夹之外。
AgentDojo unsafe 的[模型参数说明](skills/agentdojo-annotation/references/model_parameters.md)
列出了内置支持及其他模型的配置方法。

示例请求：

```text
使用 $agent3sigma-annotation，对 /path/to/cases 做 unsafe 标注，
结果写入 /path/to/new-run。先执行 dry run。

使用 $agentdojo-annotation，对 /path/to/cases 做 task-alignment，
结果写入 /path/to/new-run；无法验证的任务约束保留为待复核。
```

## 如何读结果

完整流程会生成 `quality_summary.json`、`case_outcomes.jsonl` 和运行记录。
汇总分别列出完成、明确排除、待复核和失败。例如：

```text
本轮 10 条：完成 8 条，明确排除 0 条，待复核 2 条，失败 0 条。
运行结果：需要复核。
```

以本轮筛选并应用 limit 后的案例为统计范围。dry run 不产生标注。
机器校验通过不等于标注已获得独立人工确认。

## 测试范围和已知问题

本项目有 81 项离线测试，五个模型工作流有真实 API 小样本运行记录。
AgentDojo task alignment 经过本地数据和任务约束测试。
Canary 语义判断、AgentDojo 链分类和 Agent3Sigma 历史错误元数据仍有已知限制，
详见 [KNOWN_LIMITATIONS.md](KNOWN_LIMITATIONS.md)。真实轨迹与 API 输出未打入发布包。
离线测试命令见[英文说明](README.md#local-checks)。

## 许可证

本集合的原创代码、Skill 说明、提示词模板及配套原创文档采用
[Apache License 2.0](LICENSE-CODE)。

基准数据和标注继续适用上级仓库的 `LICENSE-DATA`；第三方内容保留原有许可证
和声明，不因本代码许可证而改变。

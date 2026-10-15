# CS312 · Deep Learning Alchemy 实验笔记

与 Stanford 2026 秋季 CS312 一起训练“预测实验结果”的能力。核对日期：2026-10-15。

当前先完成 **A1 的第一轮学习率实验**：预测 → 固定配置 → 运行 → 对照曲线 → 解释偏差 → 设计留出题。我们的目标是积累可迁移的 LLM 训练经验，每轮都保留失败结果和结论边界。

**2026-10-15 状态**：完整数据已校验；H20 + SwanLab 的 32-step d8 检查通过，峰值显存 14.39 GiB；第一批 8 个正式任务已启动并接入 SwanLab，尚无最终排名。见 [第一轮实验与看板入口](reports/a1_first_batch.md)、[环境与验证记录](reports/setup.md) 和 [SwanLab 云端回读验证](reports/swanlab_h20_readback.json)。

## 目录与 Git

- `upstream/a1/`：官方 `a1_release` Git submodule，固定到 `baafc3f2d364e5ddbba75f0fe8d1cb29326920e5`。
- `sources.lock.json`：课程材料哈希、代码提交和数据 revision。
- `materials/`：官方 A1–A3 题目与首页本地快照，不提交大文件；按 lock 可重新下载。
- `experiments/a1/initial.json`：第一轮显式配置。
- `scripts/run.py`：单 GPU 入口，调用官方训练函数，上传 SwanLab 并保留本地备份。
- `notes/a1_predictions.md`：运行前的预测、信心和复盘。
- `reports/`：经过整理的小型报告；主机信息等本地配置用 `*.local.json`。
- `data/`、`runs/`、`runtime/`：忽略原始数据、权重、日志和环境，避免 Git 存储膨胀。

本仓库对应 GitHub `jiaxi98/cs312-alchemy-labs`。克隆后用 `git submodule update --init --recursive` 恢复官方代码，按 `sources.lock.json` 的哈希核对材料。官方 submodule 的 `origin` 是课程仓库；本仓库的 `origin` 是我们自己的实验仓库。

## 课程进度

以官网 2026-10-15 的公开内容为准：

| 单元 | 状态 | 我们要练的能力 |
|---|---|---|
| A1 Basics | 已发布 | 超参数、scaling 外推、seed 方差、loss 与内部统计诊断 |
| A2 Hyperparameter Scaling | 已发布 | 学习率/weight decay 随宽度、batch、数据量的变化 |
| A3 Optimization 2 | 已发布 | loss landscape、极小值、sharpness |
| A4–A5 Architecture | 官网计划，未发布题目 | 深度稳定性、容量 |
| A6 Generalization | 官网计划，未发布题目 | 数据效率；网站说明从这题转向 DNA |
| A7 Exotic | 官网计划，主题未定 | 等待题目 |

官网写明作业可使用 AI，测试重点是人能否预测新的实验结果。我们用 AI 整理代码、跑实验、做图，但预测记录留给自己先想。

## A1 路线

1. **读懂 baseline**：从 `upstream/a1/MACHINE_LEARNING.md` 到 `model_config.py → modeling.py → train.py`，说清楚数据、loss、模型与优化器。
2. **P1 超参数**：先 LR 单变量；再 warmup、batch、weight decay；根据第一轮结果选择联合 sweep 和 scheduler 对照。
3. **P2 Scaling**：按题目比较 d4–d9 的四种 recipe。先跑 d4–d7，再写下 d8/d9 预测后验证；不提前看留出尺度答案。深度和宽度在这个模型族里同时增长。
4. **P3–P4 随机性**：分开改变初始化、data order、硬件非确定性；研究微小扰动随训练放大。第一轮额外两个 seeds 只是初探，不能替代完整方差分解。
5. **P5–P6 诊断**：复用已有 loss、梯度、参数与 activation RMS；尽量不为缺日志重跑。
6. **P7 自拟预测题**：提交短小、可运行的改动和你对结果的预测。

A1 PDF 的推荐总量约 170 次训练；先完成一个可解释的小批次，再自适应选择后续实验。我们有 160 张 H20 可安排实验，实际并发数和耗时取决于当时空闲资源及单卡实测。

## 第一轮配置

正式 baseline：8 层、hidden 512、8 attention heads、SwiGLU、RoPE、pre-RMSNorm、QK norm；vocab 4096；600,000 × 1024 = **614.4M 输入 tokens**。由于 next-token shift，每条序列有 1023 个 loss targets，因此为 **613.8M 预测 targets**。训练 batch 64，默认 9375 optimizer steps。

计划有 9 项：1 次 32-step 流程检查、6 个 LR 的正式运行、2 个额外 baseline seeds。短检查改变了训练长度和 LR 日程，只验证流水线，不用于排名或外推正式结果。

```bash
python3 scripts/plan_a1.py --output experiments/a1/initial.json
python3 scripts/run.py --run-id a1-lr-0.003-s42 \
  --data-dir /path/to/data --output-dir /path/to/runs/baseline-001 --dry-run
```

在实际 GPU 环境中，每个进程只暴露一张卡：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run.py --run-id a1-smoke \
  --data-dir /path/to/data --output-dir /path/to/runs/smoke-001
```

`data-dir` 应包含 `dclm_9p6m_ctx1024/{train,val}/{tokens.bin,metadata.json}`。从锁定的数据 revision 下载全量原始序列，再由官方代码按 seed 做全局 shuffle 的前缀选择；直接下载前 600,000 条不能复现官方 seed=42 配方。

## 环境与记录口径

后续实验默认使用 **SwanLab**，项目 `jiaxi98/cs312-alchemy-labs`。首次执行 `python3 scripts/configure_swanlab.py` 配置认证；训练环境需安装 `requirements-tracking.txt`。详见 [SwanLab 接入与指标口径](docs/swanlab.md)。

官方依赖锁定 torch 2.6.0，支持 Python 3.10–3.12。已接入 H20 的 torch 2.6.0+cu124 环境；W&B 实际为 0.22.3，课程锁为 0.17.3。入口机的 CPU PyTorch 2.8.0 与远端训练环境不同。实际 GPU 检查状态和失败记录见 `reports/setup.md`。

官方训练在单卡、`num_micro_batches=1` 时启用 `torch.compile`；多卡自动走 DataParallel，梯度累积也关闭 compile。因此我们采用每卡一个独立实验。比较科学结论时固定这些实现选项，比较吞吐时记录它们。

`wandb_online=False` 会关闭官方内部诊断 logger。入口保留 `wandb_online=True`，设置 `WANDB_MODE=offline` 作为本地备份；`tracking.py` 将原始记录同步到 SwanLab，并保存完整精度的 `metrics.jsonl`。SwanLab 的 step 和本地 `completed_steps` 从 1 开始，同一步的 train/eval/RMS 合并；官方 W&B 的 optimizer step 从 0 开始。`validation.jsonl` 仍作为终端记录备份，其精度为 6 位小数。

本地入口要求全新 output 目录。失败或非有限 loss 会单独标记；不会默默跳过失败组，也不会把中断误记为完成。官方 checkpoint 可以用于后续显式续跑，当前入口不自动混入前次运行。

每轮至少保留：code commit、完整 config、数据 revision/顺序、环境、seed、train/val loss、内部 RMS、wall time 与显存。LR 实验固定参数量与 token 预算；实际 wall time 另行记录，不把它当成相同计算预算的定义。

原始来源见 `sources.lock.json`；详细 GPU 接入和当前状态见 `reports/setup.md`。

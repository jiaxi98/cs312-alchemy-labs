# A1 第一轮：学习率与 baseline 随机性

状态快照：2026-10-15T16:17:26.276851+08:00。8 个正式任务已启动，新任务当时处于数据准备或首次编译阶段；本报告不声称全部已完成。

[SwanLab 项目](https://swanlab.cn/@jiaxi98/cs312-alchemy-labs)

所有正式配置都是 d8、600,000 序列、context 1024、batch 64、9375 optimizer steps。只在指定对照中改变 peak LR 或配对的 model/data seeds。

| Run | LR | model/data seed | 快照完成步数 | 看板 |
|---|---:|---|---:|---|
| a1-lr-0.003-s42 | 0.003 | 42/42 | 7520 | [打开](https://swanlab.cn/@jiaxi98/cs312-alchemy-labs/runs/960r1pnt/chart) |
| a1-baseline-s43 | 0.003 | 43/43 | 0 | [打开](https://swanlab.cn/@jiaxi98/cs312-alchemy-labs/runs/5lgewe11/chart) |
| a1-baseline-s44 | 0.003 | 44/44 | 0 | [打开](https://swanlab.cn/@jiaxi98/cs312-alchemy-labs/runs/m0sk8hhs/chart) |
| a1-lr-0.0001-s42 | 0.0001 | 42/42 | 0 | [打开](https://swanlab.cn/@jiaxi98/cs312-alchemy-labs/runs/duknp5ch/chart) |
| a1-lr-0.0003-s42 | 0.0003 | 42/42 | 0 | [打开](https://swanlab.cn/@jiaxi98/cs312-alchemy-labs/runs/m4t7ps63/chart) |
| a1-lr-0.001-s42 | 0.001 | 42/42 | 0 | [打开](https://swanlab.cn/@jiaxi98/cs312-alchemy-labs/runs/hb4ued27/chart) |
| a1-lr-0.01-s42 | 0.01 | 42/42 | 0 | [打开](https://swanlab.cn/@jiaxi98/cs312-alchemy-labs/runs/4g4e59v7/chart) |
| a1-lr-0.03-s42 | 0.03 | 42/42 | 0 | [打开](https://swanlab.cn/@jiaxi98/cs312-alchemy-labs/runs/67s2putt/chart) |

## 记录验证

SwanLab 云端已读回真实数据：CPU 4 步的 loss/validation/RMS，H20 32 步的 loss/validation/RMS，以及运行中 baseline 的原始 W&B 历史。具体断言与浮点值见 `swanlab_readback.json` 和 `swanlab_h20_readback.json`。

baseline 开始于切换 SwanLab 之前，代码为 `caf7220`；其余 7 组代码为 `c526171`。二者模型、数据和优化循环都是同一个未修改的官方 submodule，区别是外层记录器与其依赖。baseline 的 W&B 二进制记录由独立进程跟随导入，不重跑训练。

SwanLab 日志环境增加了 SwanLab 0.10.2/protobuf 6.33.6，PyTorch 2.6.0+cu124 未变。验证通过不意味着 GPU 非确定性为零；微小 loss 差异仍需结合 seeds 判断。性能比较必须注明记录开销。

对训练效果比较使用 optimizer steps 或 tokens 横轴。导入 baseline 的云端 timestamp 是导入时间，不能用它与新实验比较训练耗时；耗时以原训练器 timing 记录为准。

这轮结束后先做完整 LR 曲线与最终 validation loss 对照，再决定联合调参范围。3 个 baseline seeds 仅为方差初探，不替代 A1 后面的完整随机性分解。

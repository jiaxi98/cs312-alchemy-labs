# SwanLab 实验记录

课程项目：`jiaxi98/cs312-alchemy-labs`。记录器版本固定为 SwanLab 0.10.2、W&B 0.22.3，见 `requirements-tracking.txt`。PyTorch 仍沿用已验证的 2.6.0+cu124。

## 认证

在控制机执行 `python3 scripts/configure_swanlab.py`。API Key 以隐藏输入读取，保存在仓库外的 `~/.config/cs312/swanlab-key`，权限 0600。项目和 workspace 保存在 Git 忽略的 `reports/swanlab.local.json`。workspace 使用账号 ID，例如 `jiaxi98`，不是显示昵称。也支持 `SWANLAB_API_KEY` 或 `CS312_SWANLAB_KEY_FILE`。

集群入口把 key 经 SSH 标准输入传输，并只注入训练进程的环境变量；不会把 key 写入启动命令、实验 config、部署快照或 Git。项目默认私有。

## 新实验

`scripts/run.py` 和 `scripts/cluster.py run` 默认使用 `--tracker swanlab-online`。显式的 `--tracker swanlab-offline` 用于没有网络时先写离线数据；旧的 `--tracker wandb-offline` 保留以便复现实验。

在线实验创建失败会报错，不静默切成另一个记录平台。在线运行同时保留本地 SwanLab、W&B 和 JSONL 备份。

| 记录 | 含义 |
|---|---|
| `train_loss` / `val_loss` | 官方训练器原始数值；validation 不通过终端打印截取 |
| `next_learning_rate` | 官方在 scheduler.step 后记录的 LR，即下一步 LR |
| `completed_steps` | 已完成 optimizer steps，从 1 开始；也是 SwanLab 显式 step |
| `input_tokens_seen` | 完成步数 × batch size × context；当前计划各配置无不满 batch |
| `logging/rms/...` | 官方 module activation、parameter、pre-clip gradient RMS |
| `diagnostics/nonfinite/...` | 指标非有限时记录 1，原 loss 在 JSONL 为 null，不伪造数值 |

官方每次评估会另发一条 W&B log；其隐式 record index 和 optimizer step 不同。`scripts/tracking.py` 把同一 optimizer step 的 train/eval/RMS 合并，统一以 `optimizer_step + 1` 为横轴，防止不同评估频率造成横轴错位。官方学习与优化代码没有修改。

配置保留原始 hyperparameters、实际解析后的模型与训练配置、代码提交、数据 revision、环境及任务状态。SwanLab 系统监控提供利用率和显存等时序；最终 peak allocated memory 另由 PyTorch 记录。

## 已运行基线

`scripts/import_wandb.py` 直接读取官方 W&B 二进制记录，保留完整浮点精度及内部 RMS。`--follow` 可跟随尚在运行的基线；最后一个未写完的 step 等待下次记录后再提交，避免漏掉同一步的 validation。它不会改变、暂停或重启原训练。

```bash
python scripts/import_wandb.py --run-dir /path/to/a1-baseline-001 \
  --project cs312-alchemy-labs --workspace jiaxi98 --follow
```

## 验证

`python -m unittest discover -s tests -v` 覆盖 train/eval 同步、下一步 LR 标签、非有限值、真实 SDK 写入后的 step 与 RMS 记录。测试读取 `.swanlab` 的真实持久化数据，避免只检查函数调用成功。

官方文档：

- https://docs.swanlab.cn/api/py-init.html
- https://docs.swanlab.cn/api/py-login.html
- https://docs.swanlab.cn/guide_cloud/integration/integration-wandb.html

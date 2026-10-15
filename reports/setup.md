# 首次环境接入记录

记录日期：2026-10-15。课程代码固定到 `baafc3f2d364e5ddbba75f0fe8d1cb29326920e5`。

## 已验证

- 本仓库使用 Git；官方代码是 `upstream/a1` submodule。clone 后执行 `git submodule update --init --recursive`。
- GPU 节点环境：NVIDIA H20，Python 3.12.6、PyTorch 2.6.0+cu124、CUDA 12.4。W&B 0.22.3，与课程锁定的 0.17.3 不同；离线记录已通过实际检查。
- 课程完整训练数据 19,660,800,000 bytes，validation 2,048,000 bytes。两份 token 文件及两份 metadata 全部通过发布者 manifest 的 SHA256；数据 revision 见 `sources.lock.json`。
- 4-step 合成 CPU 集成检查完成，逐步 loss、两次 validation、activation/gradient/parameter RMS 均记录成功，见 `cpu_check.json`。此结果只证明流水线贯通。

## 首次 H20 检查及修正

`h20-smoke-001` 在第一步的 `torch.compile` 阶段失败：共享盘上新生成的 Triton `cuda_utils.so` 在动态加载时报告 `Stale file handle`。该次检查没有完成训练，不作为课程结果。

运行入口已将 TorchInductor 与 Triton 的编译缓存放在节点本地 `/tmp`，且在 import torch 前设置路径。日志和 checkpoint 仍保存到配置的持久存储。`CS312_CACHE_ROOT` 可以指定其他节点本地目录；每次运行的实际 cache 路径会写入 `environment.json`。

部署改用代码快照和逐文件哈希，不携带 Git submodule 的 `.git` 指针。快照记录课程仓库 commit、官方代码 commit，每次运行前校验所有部署文件。只有已提交且工作区干净的版本可部署。

`h20-smoke-002` 已通过：同一 d8、context 1024、batch 64、BF16 autocast 和 `torch.compile`，完成 32/32 steps；32 条逐步 loss、2 条 validation 及 RMS 日志均写入。总耗时 175.51 秒，峰值 allocated 显存 14.39 GiB。首次 step 耗时 93.90 秒，包含编译；短 run 的总时间不能用来估计稳定训练吞吐。可复核记录见 `h20_smoke.json`。

正式 baseline 已按同一冻结代码版本启动：`a1-baseline-001`，600,000 序列、9375 steps、LR=0.003、model/data seed=42。随后加入 5 个 LR 对照及 baseline seed 43/44，第一批合计 8 个正式任务。当前尚无最终 validation loss；短检查结果不作为正式实验结论。见 `a1_first_batch.md`。

SwanLab 0.10.2 已通过真实云端验证。新的日志环境独立安装 SwanLab/protobuf 6.33.6，保留 PyTorch 2.6.0+cu124；既有 baseline 继续原环境，由独立导入进程跟踪其 W&B 二进制文件。H20 新日志环境 32-step 检查通过并读回了 32 个 train loss、2 个 val loss 和梯度 RMS，见 `swanlab_h20_readback.json`。

已从 GitHub 重新 clone 并初始化 submodule，确认官方 checkout 提交与 lock 一致、dry-run 入口可用。

## 本地或其他 GPU 环境

沿用官方 `upstream/a1/pyproject.toml`、`uv.lock` 安装依赖；项目支持 Python 3.10–3.12。下载器和计划生成器只需 Python 标准库。

```bash
python3 scripts/prepare_data.py --output-dir /path/to/data/dclm_9p6m_ctx1024
CUDA_VISIBLE_DEVICES=0 python scripts/run.py --run-id a1-smoke \
  --data-dir /path/to/data --output-dir /path/to/runs/smoke-001
```

下载完成后会生成 `READY.json`，其中记录 revision、manifest 哈希和各文件 SHA256。正式运行要求此记录存在、revision 一致、文件大小匹配。

## 当前集群适配

`scripts/cluster.py` 复用现有 fleet adapter 的实例查询及 SSH 连接。平台凭据由 adapter 在内存中读取；本仓库不保存凭据。适配配置保存在被 Git 忽略的 `reports/cluster.local.json`：

```json
{
  "fleet_scripts": "/path/to/existing/fleet/scripts",
  "remote_root": "/persistent/storage/cs312",
  "remote_python": "/path/to/cuda/environment/bin/python",
  "http_proxy": "http://your-proxy:port"
}
```

也可通过 `CS312_FLEET_SCRIPTS`、`CS312_REMOTE_ROOT`、`CS312_REMOTE_PYTHON`、`CS312_HTTP_PROXY` 覆盖。控制机需安装 paramiko，GPU 节点使用已经匹配 CUDA 的训练环境。

```bash
python3 scripts/cluster.py stage
python3 scripts/cluster.py download
python3 scripts/cluster.py run --attempt h20-smoke-002 --gpu 0
python3 scripts/cluster.py status
```

一次 run 对应一个可见 GPU。启动前检查卡占用并持有启动锁；attempt 名唯一，已有 attempt 先检查状态，避免重复启动。当前 status 以选中的 host 为准。

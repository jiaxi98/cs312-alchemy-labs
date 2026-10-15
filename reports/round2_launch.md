# A1 第二轮：160卡并行实验

启动核对：2026-10-15T17:55:22.902102+08:00。**20台 × 8张H20，160个正式任务已进入训练并产生validation loss**。GPU进程探测覆盖全部160卡，利用率中位数98%。这是一个时间点的实测，不表示整个队列运行期间始终满卡。

[SwanLab 项目](https://swanlab.cn/@jiaxi98/cs312-alchemy-labs)

## 这轮研究的问题

| 部分 | 去重后配置数 | 内容 |
|---|---:|---|
| P1 | 141 | warmup、weight decay、batch×LR、weight decay×LR、warmup×LR、scheduler和momentum |
| P2 | 32 | d4–d7 × 4种官方recipe × 2 seeds；先保留d9不跑 |
| P3 | 48 | 初始化/数据/共同seed变化、不同H20节点重复、配方改变后的方差 |
| P4 | 12 | 6 seeds × 单token微扰/control |
| 合计 | 233 | 每配置一张卡，614.4M输入tokens |

详细配置见 `experiments/a1/round2.json`；事前假设和实验边界见 `notes/a1_round2_predictions.md`。P5/P6复用本批曲线和RMS，不额外重复同一训练。

## 执行方式

每台节点的8个worker分别持有GPU锁，从共享目录原子领取一个任务。任务结束后领取下一项，不等待同机其余GPU。失败写入独立result，保留日志，然后补上下一项。原有163次尝试包含160个在训练任务和3个启动失败，尚有70个主队列配置待领取。

所有节点使用PyTorch2.6.0+cu124、SwanLab0.10.2、W&B0.22.3。完整数据复制到本地并验证四个文件SHA256；数据打乱和前缀选择仍使用官方代码。编译cache也保存在本地。节点实测最大SM clock供编译器估算使用，避免160个进程反复查询物理GPU0。

## 检查与修正

8项真实GPU检查通过：batch256含不满尾batch、dropout、deterministic、token perturb、constant、cos、WSD、普通d8。batch256的9步检查处理2050条序列，末步tokens被准确封顶2050×1024，peak allocated memory56.19GiB。见 `round2_checks.json`。

主队列3个β1=0的配置在创建AdamW时失败，原因是JSON整数0不满足PyTorch的float beta要求，尚未产生训练步。已改成0.0，并以 `a1-r2-133-fix1`、`a1-r2-134-fix1`、`a1-r2-135-fix1` 单独排队。修正worker等待同一GPU锁释放，最多仍使用现有160卡。原失败记录保留，不能当成数值发散。

修正队列第一次启动误用旧部署快照，因缺少repair plan立即退出、未领取GPU；第二次已经核对新快照并确认3个worker处于waiting_for_gpu。运行中的主队列从未改写或重启。

主队列冻结代码为 `65f35da`，修正版本为 `9e81553`。仓库里的round2配置已修正beta类型；旧尝试仍通过其原始commit和部署哈希复现。

## 验证依据

`round2_launch.json` 记录每台节点的CUDA进程卡数和利用率；`round2_cloud_samples.json` 记录各类实验从SwanLab读回的train/val loss。核对时160个活跃任务全部至少完成780个optimizer steps且都有validation，尚未比较终值。

## 后续阅读

在SwanLab按名称前缀 `P1 batch×lr`、`P1 schedule`、`P2 d`、`P3`、`P4` 筛选对应主题；对比batch时按 `input_tokens_seen` 对齐，因为每个batch对应的optimizer steps不同。

P2的d9保持未运行，待d4–d7结果支持一个明确预测后再释放。默认d8结果此前已见，不称为盲测。P3全用H20，P4只改变初始时刻的一个token，各自尚未覆盖整题的全部拓展。

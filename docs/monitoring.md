# A1 后台监控与自动接续

监控每2分钟运行一次。节点上的现有worker在当前任务结束后立即领取下一个任务；控制机监控核对任务状态、进程、训练步数、GPU占用和GPU锁，必要时恢复空闲卡上尚未领取的工作。

## 当前范围

- 仅使用 `allocation_policy.json` 允许的 **box2 + hy，最多80张H20**。
- H3_grpo和box_flow_grpo_1_bj已经预留，不会被监控查询或启动课程任务。
- 跟踪207项有效A1补充目标，迁移前的103条记录由迁移任务替代，不重复计数。
- 固定GPU实验按指定host/GPU继续串行，其余任务并行接续。
- 只执行已声明的A1计划；A2或新的开放课题不自动启动。

## 恢复规则

仅当GPU没有CUDA进程、课程GPU锁空闲、任务从未领取、host和GPU符合配置时才恢复worker。恢复前再次核对，锁随新worker启动继承，避免检查和启动之间的抢占窗口。

已经开始但失败的任务不自动重复运行；数值发散作为实验结果保留。进程消失、超过30分钟无训练进度、主机不可达等写入异常记录。数值发散不会被当作“继续重试直到成功”的理由。

每次恢复使用固定部署版本，并校验计划SHA256。每轮恢复最多10个空闲GPU，其余在下次检查继续；已有正常worker不受影响。运行中的任务不会因为监控启动而重跑。

## 查看和控制

- 实时摘要：`reports/monitor_live.local.md`。
- 机器可读状态：`reports/monitor_status.local.json`。
- 事件：`runtime/monitor/events.jsonl`，仅新增完成、异常、恢复动作和总完成时记录。
- 监控日志：`runtime/monitor/cron.log`。
- 全部目标完成：生成 `runtime/monitor/COMPLETE.json` 与 `reports/monitor_final.local.json`，之后定时入口不再轮询GPU。计划变化后可继续检查新目标。

创建 `runtime/monitor/STOP` 会停止监控检查和恢复，**不会停止已经启动的训练或节点worker**。要撤出某个计算任务，应先明确更新资源规则并停止对应课程进程。

当前聊天环境没有可调用的自动唤醒工具，因此使用控制机crond后台执行，不承诺无人值守时向聊天主动发消息。实验曲线仍由SwanLab持续更新，监控摘要保存在上述本地文件。

## 安装与复现

使用控制机持久目录里的独立Python环境安装 `requirements-monitor.txt`。凭据继续读取仓库外的受限文件，不进入命令参数、cron文本或Git。

1. 提交代码并通过 `scripts/cluster.py stage` 部署。
2. 将该次 `reports/stage.local.json` 复制为 `runtime/monitor/stage.local.json`，固定恢复所用版本。
3. 执行 `python scripts/monitor.py --apply` 验证一次完整检查。
4. 执行 `python scripts/install_monitor.py --python /path/to/persistent/venv/bin/python`。

安装器先备份crontab，只替换带 `cs312-a1-monitor-80gpu` 标记的行。单次检查有110秒总超时和文件锁，重叠触发会跳过。原有系统或其他项目的cron任务保持原样。

# 一次性失败与执行边界审计

TASK：LING_KDA_ROTATION_DUAL_AXIS_STABLE_RECOVERY_V1。科学协议继续 LING_KDA_ROTATION_DUAL_AXIS_FUNCTIONAL_V1，六篇文档及五条件顺序、8448 consumed tokens、256 fullV FP32 KL、rotation/codec/recurrence/kernel/dtype 均冻结。本文在CPU开发阶段形成，不能作为GPU资格或科学结论。

## 本科学任务的两个停止

1. 首执行：三源实际保存外部compute PID3600300、UID1008/start55595272，896MiB。已证实并发资源冲突，首失败SHA fdb6ff1ddcab5b2f319fbc1e5a184cf0291cbaad1f209c242014ba79bbba735f，不能改称OOM或显存余量不足。
2. 续执行：H_R128 fresh rc1 / OWN_PARENT_FRESH64_EXACT_MISMATCH；首失败SHA 35f6484d87fd9134db4078942454912301c6a1450f994fa2d318c1825cd2d292。286点已存raw无外部compute、查询UNKNOWN或2GiB余量不足。8256 parent integrity返回活动after_anchors字典与proof列表，parent继续8448后污染原参考；标量QDQ写入仍148608，anchor计数变115200/4608，而8256应111744/1152。独立fresh logits/cache/audit/RNG hash未保存，数值是否一致仍UNKNOWN。另一个独立比较表示缺陷是JSON会将整数键变成字符串，直接与fresh内存整数键比较也不等。两者均是证据比较层问题，不能据此宣称GPU实际数值PASS。

## 其他历史故障独立分型

旧GPU3 material-memory/empty-PID为进程可见性或非原子生命周期证据不一致；不能归因外部。旧GPU6四文档32K的free2025193472字节确低于2GiB116.625MiB，owner唯一、查询正常，是真实余量不足，非CUDA OOM/抢卡。更旧4090缺import是启动工程错误；旧4090 assess先于raw落盘导致首失败raw缺失，归属UNKNOWN；PYTHONHASHSEED初始化与早期恢复exact失败保留原分类。旧CPU封存误解析nested时间字典是STOP后的独立报告缺陷，不能合并为GPU根因。各旧目录与seal保持只读。

## 恢复与观测顺序

原CPU checkpoint在pos8191原model_step→原QDQ/writeback之后保存next_position8192、完整hybrid cache、audit.previous/计数/proof及Python/NumPy/CPU Torch/CUDA RNG。fresh验证完整binding/SHA/cursor后复制cache和audit，恢复四类RNG，再从8192运行；新scoring_cursor及scored_positions明确阻止重复已保存评分。原18层QDQ计数与next-kernel消费检查不变。

8256参考现在是独立typed值树、tensor独立CPU副本与内容SHA raw blobs，递归保留int/str键、tuple/list、dtype/shape/cursor。原子独占发布不覆盖，8256之后继续8448必须核文件字节SHA不变。fresh每token先独立保存全部数值证据，再逐字段与同cursor parent比较，首次差异独占保存并停止；无缺字段容忍。cache、actual QDQ codes/scales、实际writeback/initial_state、audit/proof、四类RNG与完整logits均在64步中核验。CPU回归与实际cache/RNG前后SHA检查观测不改变状态；证据不能被装回模型作为state。

## 三个边界及控制复杂度

Scientific Worker继续原模型一步/原codec/原KL，只添加被动证据适配器和显式同cursor比较；make_audit数学函数整体AST以及所有核心文件字节必须不变。Minimal Coordinator沿用一个顺序Popen路径、一把锁和原三源资源判据，负责完整生命周期与账本。Offline Verifier只在CPU端复核合法EXIT/PROFILE/输入/UUID/typed证据/SHA及结果；时间字典只作时间，不能当显存snapshot。不建立第二套launcher或修改KDA/FLA。

先真实独立canary128步FP→H/L6免量化Value侧坐标关系，再五条件各自身0→8448及8192CP fresh64。Key对角gate与任意旋转不假定交换；当前路径是Value侧codec后逆变换回native。未量化roundtrip存在冻结矩阵与FP32舍入差，记录实际logit差和实际state坐标算术关系，不借原fresh64 exact容差解释native-vs-roundtrip。

## 资源、预算与终态

只授权一张3090，初选GPU6 UUID GPU-de53f243-b271-521c-d875-d96611ed8574，GPU启动前仍须实际三点空闲门禁；所有新资格同UUID，无旧CP/PASS继承。2GiB运行余量、三源foreign/UNKNOWN、UID/start/UUID、原admission与终态三点均不放宽。UNKNOWN保持FAIL_CLOSED，不伪称外部。Qwen计算位于独立.55任务，Ling在.29，禁止自动操作Qwen，启动时还要只读确认本UUID空闲。

历史累计1.2779262566580696 GPUh由两份封存账本绑定，16硬上限剩14.72207374334193，15.5保护线剩14.22207374334193。新综合GPU资格控制2.5 GPUh以内并累计；只有真实五条件成本、1.25原安全系数、六篇正式预测加0.25退出余量全部满足才正式。现仅有FP约22.21分钟和H_R128约24.03分钟历史完整速率，不能宣称30条预算可行。资源/真实数值失败只保存首失败停止，不切卡、修改容差或开启下一轮GPU调试。

CPU完整回归、真实GPU资格、第一篇结果、六篇完整科学结果分开报告。若失败或预算不足，只交付实际证据与NOT_RUN，所有科学结论缺失不得填0。最终停止，不自动新实验。

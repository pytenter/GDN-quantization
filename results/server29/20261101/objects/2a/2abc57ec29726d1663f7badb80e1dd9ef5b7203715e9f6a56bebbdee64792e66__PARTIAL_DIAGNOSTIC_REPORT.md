# HMMT26 首次正式执行故障诊断（CPU恢复）

状态：PARTIAL_WITH_VALIDATED_ARTIFACTS；GPU执行已停止，未自动重试。

首因是新增C0审计记录的JSON整数键序列化缺陷。NativeFPAdapter.audit返回helper_calls使用整数层号；写盘将键转换为字符串，重新计算摘要时排序从数字顺序变成字典序。对原始完整RESULT只恢复该字段的整数键，恰好得到原保存摘要，已在CPU精确验证。未改任何原始输出、源、账本或seal。

真实运行：首个C0(q01)生成11728 tokens并EOS，原owning Popen整数0、ownNVML已释放；控制器在结果摘要校验断言处停止，controller/parent/Bashwait均1。其余C1(q01)、C3(q01)、C0(q02)原owner真实退出-15并释放，全部中断前缀计分credit0。原controller完成并持久化计数0；独立CPU检查可以完整保全1条原始完整C0轨迹，不把它冒充原controller已接受的99条结果。

没有99条完整结果，未评分，不生成全量accuracy或空科学CSV。Hadamard C2仍因预冻结完整模型数值门禁失败未运行，这与本次CPU序列化错误分别记录。已有吞吐检查是停止前的真实观测，不表示当前GPU仍在运行。

独立修复候选只将C0 audit.helper_calls键转为字符串；其余85个冻结文件逐字相同，量化、mask、采样、模型、State公式及快速Direct kernel不变。候选未冻结许可、未启动GPU。CPU实际方法和完整原payload写盘往返测试记录另附。

原本地终态helper已claim后因只支持Bashwait0而拒绝这次真实Bashwait1，claim/FIRST_FAILURE保留不重复。第一独立CPU恢复又在SSH认证前失败，未执行远程Python，保留其claim和firstfailure；只恢复SSH连接后使用另一个明确独立CPU恢复目录，未动GPU。

后续：当前跟进明确禁止自动GPU重试。独立修复版本的新GPU执行需要明确授权；在此之前保持停止，不热改、不重启旧owning链。完整旧轨迹应保留，恢复时须逐项验证原source/model/prompt/seed/token/退出/释放和独立摘要因果证明，不按答案决定是否重跑。

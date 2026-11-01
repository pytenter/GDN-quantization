首次工程失败：原 numerical_recovery.compare AST 的 tsha 全局未恢复；独立 CPU 复现见原因审计。CPU 失败结果未拦截 GPU dispatch 是本次执行错误。发现后立即终止、保留旧源码和监督器 first-failure；未在本 attempt 中修复或重跑 GPU。

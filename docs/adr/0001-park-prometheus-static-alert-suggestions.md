---
status: accepted
---

# 暂停 Prometheus 静态告警规则推荐

Prometheus alert rule 推荐的产品行为与安全边界尚未确定，因此移除 `Static alert suggestions` 及其 PromQL 目录、导入导出工具、审核流程、持久化模型和测试。讨论、调研、设计或执行计划均不构成实现授权；除非项目所有者在当前请求中明确授权实现，否则必须保持该能力不存在。获得授权后，也必须先确认需求与安全模型，再添加代码。

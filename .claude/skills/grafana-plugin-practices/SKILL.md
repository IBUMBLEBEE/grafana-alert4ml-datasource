---
name: grafana-plugin-practices
description: Grafana datasource 插件仓库级规则。当修改 plugin.json、前端查询编辑器、datasource 行为、健康检查、日志、配置处理或兼容性相关代码时使用。
---

# Grafana 插件开发规范

本技能定义将 Alert4ML 作为 Grafana datasource 插件维护时必须遵守的仓库级规则。

## 何时使用

- 修改 src/ 下影响插件 UI、查询编辑器、配置编辑器或 datasource 前端行为的文件
- 修改 pkg/plugin/ 或 pkg/sdk/ 下影响后端查询处理、健康检查、资源接口、日志或插件行为的文件
- 更新 src/plugin.json、provisioning、tests 或兼容性行为

## 必查文件

- src/plugin.json
- src/datasource.ts
- src/components/QueryEditorv2.tsx
- pkg/plugin/datasource.go
- README.md
- tests/

## 仓库规则

1. 必须首先把这个代码库视为 Grafana datasource 插件，其次才是 ML 产品。插件安全性、兼容性和可运维性优先于花哨的 UI 或后端行为。
2. 必须保留后端 datasource 架构。当前仓库已经使用 backend plugin，涉及后端敏感能力的工作必须留在 Go 和 Rust 中，不能推到浏览器侧。
3. 前端代码必须遵守 Grafana UI 约定。优先使用 @grafana/ui 组件和 Grafana 主题模式，而不是自定义 UI 体系。
4. 禁止在前端产物中保留 console logging。浏览器侧调试输出不能进入生产代码。
5. 禁止在前端状态、查询载荷、日志、截图或文档中暴露 secrets、tokens、DSN 或凭据。敏感配置必须放在 secureJsonData 或纯后端配置路径中。
6. 禁止通过插件输入执行任意代码。查询参数、宏和模型参数都必须是经过校验的数据，而不是可执行代码。
7. 必须保持插件元数据准确。如果行为变更影响插件能力，需要确认 src/plugin.json 中 backend、metrics、logs、annotations、alerting 和 grafanaDependency 等声明仍然正确。
8. 优先采用渐进式 UX。高级 ML 选项应该可发现，但不能压垮默认查询流程。
9. 对隐藏或空查询必须跳过，而不是执行无意义请求。
10. README 和面向运维的行为说明必须与实际实现保持一致。插件配置应该可理解，而不是要求用户反向阅读源码。

## 兼容性规则

1. 必须假设运行时兼容性问题是真实存在的。Grafana 插件即使编译成功，也可能因为 Grafana 替换前端依赖而在运行时失败。
2. 引入与 Grafana 版本相关的行为时，优先使用运行时守卫，而不是分叉仓库维护多套实现。
3. 在可行情况下维持一条受支持的代码路径，并与 src/plugin.json 中声明的 grafanaDependency 范围保持一致。
4. 如果改动影响 UI、datasource 执行或插件元数据，必须至少做一步面向兼容性的验证，而不能只依赖静态阅读。

## 运维规则

1. 健康检查是插件契约的一部分。后端改动必须保留或提升可诊断性。
2. 后端日志必须使用明确的级别。错误条件记录为 error，常规诊断优先使用 debug，日志中绝不能泄漏敏感值。
3. 禁止给插件运行时行为加入本地文件系统假设。插件实例共享环境，正常运行不能依赖本地文件。
4. 当 datasource 实例设置或插件配置才是正确归属时，禁止引入依赖环境变量的 datasource 行为。

## 变更工作流

1. 先判断改动影响的是前端 UX、后端查询执行、插件元数据，还是兼容性表面。
2. 在对应层完成实现更新。
3. 重新核对插件元数据和面向运维的行为说明。
4. 当用户可见行为变化时，同步更新 README 或文档。
5. 执行最小但有效的验证集合。

## 最低验证要求

- npm run typecheck
- npm run build
- go test ./pkg/...
- 如果查询编辑器或插件行为发生变化，考虑执行 npm run e2e 或针对 Grafana 的定向 smoke 验证。

## 反模式

- 在前端代码中保留 console.log
- 不遵循 Grafana UI/主题模式，而是硬编码视觉 token
- 通过前端查询对象传递 secrets
- 插件能力已经变化，却仍把 plugin.json 当成静态样板文件
- 添加依赖任意用户代码或本地文件系统访问的插件行为
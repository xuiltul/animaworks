# 工程师专业指南

## 编码原则

- **最小变更**：对现有代码的修改应保持在必要的最小范围内。大规模重构仅在明确指示时进行。范围外的修正另行记录为任务
- **YAGNI**：不要为预测未来扩展而使代码复杂化。抽象化应在同一模式出现3次后进行（Rule of Three）
- **安全性**：必须进行输入验证、SQL使用参数绑定、禁止硬编码密钥、使用`pathlib.Path`防止路径穿越、规避`shell=True`

## 代码质量

- 必须使用`from __future__ import annotations` + `str | None` 形式的类型提示
- 使用`pathlib.Path`进行路径操作、Google-style docstring、`logging.getLogger(__name__)`
- 使用 Pydantic Model / dataclass 定义数据
- 语义化提交：`feat:` / `fix:` / `refactor:` / `docs:` / `test:` / `chore:`

## 测试・错误处理

- 代码修改后确认相关测试。为新函数添加单元测试
- 捕获具体的异常（禁止裸`except:`）。重试时使用指数退避
- `async/await` + `asyncio.Lock()`。CPU-bound使用`asyncio.to_thread()`

项目固有规范请参照仓库中的`.cursorrules` / `CLAUDE.md`

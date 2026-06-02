# Scripts — MOSS 集成验证与探索脚本

可独立执行的 Python 脚本，不属于 pytest 套件，不打包到 `ghoshell_moss` 包中。

## 约定

- 自包含：每个脚本独立可执行，`python scripts/<area>/<name>.py` 即可
- 不依赖 pytest：用 `assert` 和 `print`，不用 `unittest`/`pytest`
- 清晰输出：通过则 `OK`，失败则打印 traceback
- 按功能域分子目录（如 `ghost/`、`channel/` 等）

## 批量运行

```bash
# 单个
python scripts/ghost/test_run_ghost_instance.py

# 按目录
for f in scripts/ghost/test_*.py; do echo "=== $f ===" && python "$f" && echo "OK" || echo "FAIL"; done
```

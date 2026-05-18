"""ghost 发现体系验证 — all_ghosts() 懒加载 + CLI list/show 的底层路径."""
from ghoshell_moss.host import Host
from ghoshell_moss.core.blueprint.ghost import GhostMeta

host = Host()

# 懒加载 — 第一次调用触发扫描
ghosts = host.all_ghosts()
assert isinstance(ghosts, dict)
assert len(ghosts) >= 1, f"expected at least mock ghost, got {list(ghosts.keys())}"

# 每个 value 都是 GhostMeta 实例
for name, meta in ghosts.items():
    assert isinstance(meta, GhostMeta), f"{name}: expected GhostMeta, got {type(meta)}"
    assert meta.name() == name
    assert isinstance(meta.prototype(), str)
    assert isinstance(meta.description(), str)
    print(f"  {name} ({meta.prototype()}): {meta.description()[:80]}")

# 按名查找
mock = ghosts["mock"]
assert mock.name() == "mock"
assert mock.prototype() == "MockGhost"

# GhostMeta 协议完整性
assert isinstance(mock.nuclei_metas(), list)
assert isinstance(mock.providers(), list)
assert mock.contracts() is not None
assert isinstance(mock.identifier, str)

print(f"OK — {len(ghosts)} ghost(s) discovered")

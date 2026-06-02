"""Verifies observability hooks: on_articulate_exit, inspect_state, inspect_loop_health.

Tests ghost-developer hook override (MockGhost) and GhostRuntime diagnostics
for the REPL/script consumer — both without monkey-patching.
"""
import asyncio
from ghoshell_moss.host import Host
from ghoshell_moss.ghosts.mock import MockGhostMeta, MockGhost
from ghoshell_moss.ghosts.mock._runtime import MockArticulator
from ghoshell_moss.core.blueprint.ghost import Ghost
from ghoshell_moss.core.blueprint.mindflow import Moment

host = Host()
meta = MockGhostMeta(name="test_hooks")

# Pre-check on meta
assert meta.name() == "test_hooks"
print("meta.name OK")

gr = host.run_ghost(meta)


async def main():
    async with gr:
        ghost = gr.ghost
        assert isinstance(ghost, MockGhost)

        # ── 1. inspect_loop_health before any articulation ──
        health = gr.inspect_loop_health()
        print(f"loop_health: {health}")
        assert set(health.keys()) == {"main", "articulate", "action"}

        # ── 2. articulate + hook (simulating GhostRuntime pattern) ──
        art = MockArticulator(moment=Moment())
        ghost.set_articulate_responses(["hello", " world"])
        logos_parts = []
        async for delta in ghost.articulate(art):
            logos_parts.append(delta)
        logos = "".join(logos_parts)
        assert logos == "hello world"

        # GhostRuntime calls this after each articulate cycle.
        # Ghost developers test it manually as shown here.
        ghost.on_articulate_exit(art, logos, error=None)
        assert ghost._last_logos == "hello world"
        assert ghost._last_error is None
        print("on_articulate_exit OK — last_logos captured")

        # ── 3. ghost.inspect_state() ──
        state = ghost.inspect_state()
        print(f"inspect_state: {state}")
        assert state["last_logos"] == "hello world"
        assert "articulate_responses_remaining" in state

        # ── 4. Ghost ABC contract ──
        assert hasattr(Ghost, "on_articulate_exit")
        assert hasattr(Ghost, "inspect_state")
        print("Ghost ABC hooks present")

        gr.close()

    # ── 5. inspect_loop_health after close ──
    health_after = gr.inspect_loop_health()
    print(f"loop_health after close: {health_after}")

    print("OK — all observability hooks verified")


asyncio.run(main())

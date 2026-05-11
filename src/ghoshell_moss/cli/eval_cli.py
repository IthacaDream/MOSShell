"""
moss eval — execute arbitrary Python code in the live MOSS runtime.

Uses Compiler + Executor from core.codex to run code in an isolated module,
capturing stdout and return value as structured JSON.
"""

import typer
import json
import sys
import traceback
from pathlib import Path
from typing import Optional

from ghoshell_moss.core.codex.compiler import Compiler
from ghoshell_moss.core.codex.executor import Executor, ExecutionResult
from .utils import console, print_error


def _execute(
    code: str,
    module_path: Optional[str] = None,
) -> ExecutionResult:
    """Compile and execute code, optionally in a module's context."""
    origin = None
    if module_path:
        import importlib
        try:
            origin = importlib.import_module(module_path)
        except ModuleNotFoundError:
            raise ValueError(f"Module not found: {module_path}")

    if origin:
        executor = Executor(origin=origin)
    else:
        import types
        executor = Executor(origin=types.ModuleType("__moss_eval__"))

    return executor.execute(code)


def _serialize_result(result: ExecutionResult) -> dict:
    """Serialize execution result for JSON output."""
    returns = result.returns
    try:
        _ = json.dumps(returns)
        serialized = returns
    except (TypeError, ValueError):
        serialized = repr(returns)

    return {
        "returns": serialized,
        "std_output": result.std_output,
    }


def eval_code(
    code: str = typer.Argument(default="", help="Python code to execute"),
    module: Optional[str] = typer.Option(None, "--module", "-m", help="Import path of module to use as execution context"),
    file: Optional[Path] = typer.Option(None, "--file", "-f", help="Read code from file instead of argument"),
):
    """
    Execute Python code in the live MOSS runtime environment.

    Code runs in an isolated module with full access to the host process.
    stdout is captured automatically. Assign to __result__ to return a value.

    Examples:
        moss eval "print(type(host))"
        moss eval --module ghoshell_moss.host.runtime "print(host.mode.name)"
        moss eval --file debug_script.py
    """
    if file:
        try:
            code = file.read_text()
        except FileNotFoundError:
            print_error(f"File not found: {file}")
            raise SystemExit(1)

    if not code.strip():
        print_error("No code to execute. Provide code as argument or via --file.")
        raise SystemExit(1)

    try:
        result = _execute(code, module_path=module)
    except Exception:
        print_error(f"Code execution failed:\n{traceback.format_exc()}")
        raise SystemExit(1)

    output = _serialize_result(result)
    console.print(json.dumps(output, ensure_ascii=False, indent=2))

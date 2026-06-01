from pathlib import Path
from typing import List

from rich.console import Console
from rich.table import Table
from rich.syntax import Syntax
from rich.panel import Panel

from .utils import console, print_simple_table, print_simple_panel
import typer

from ghoshell_moss.host import Host
from ghoshell_moss.core.codex.discover import ScanError

mode_app = typer.Typer(help="Manage MOSS Host Modes (Environment Isolation).", no_args_is_help=True)

# Manifest 文件约定 — mode 目录下可选的声明文件
_MANIFEST_FILES = [
    ("channels.py",  "main channel 入口"),
    ("providers.py", "IoC Provider 声明"),
    ("configs.py",   "配置模型声明"),
    ("topics.py",    "事件协议声明"),
    ("resources.py", "资源存储声明"),
    ("nuclei.py",    "感知核声明"),
    ("contracts.py", "mode 专属契约"),
]


def _display_scan_errors(errors: list[ScanError]) -> None:
    """显示发现过程中的非致命 scan 错误。"""
    if not errors:
        return
    console.print(f"\n[yellow]Warning: {len(errors)} mode discovery error(s):[/yellow]")
    error_data = [
        [err.module_path, err.stage, f"{type(err.exception).__name__}: {err.exception}"]
        for err in errors
    ]
    print_simple_table(
        data=error_data,
        headers=["Module", "Stage", "Error"],
    )


def _list_manifest_files(mode_dir: Path) -> list[tuple[str, str, bool]]:
    """检查 mode 目录下的 manifest 文件存在状态。返回 (filename, description, exists)。"""
    result = []
    for filename, desc in _MANIFEST_FILES:
        exists = (mode_dir / filename).is_file()
        result.append((filename, desc, exists))
    return result


def _print_manifest_files_human(manifest_status: list[tuple[str, str, bool]]) -> None:
    """人类模式：显示 manifest 文件清单，带 [present]/[not found] 标记。"""
    rows = []
    for filename, desc, exists in manifest_status:
        status = "[green]present[/green]" if exists else "[dim]not found[/dim]"
        rows.append([f"  {filename}", status, f"[dim]{desc}[/dim]"])

    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column("File", style="bold")
    table.add_column("Status")
    table.add_column("Description", style="dim")
    for row in rows:
        table.add_row(*row)
    console.print(table)


def _print_manifest_files_ai(manifest_status: list[tuple[str, str, bool]]) -> None:
    """AI 模式：纯文本 manifest 文件清单。"""
    for filename, desc, exists in manifest_status:
        status = "present" if exists else "not found"
        console.print(f"  {filename}  ({status})  — {desc}")


@mode_app.command(name="list")
def list_modes():
    """
    List all discovered modes in the current MOSS workspace.
    """
    host = Host()
    modes = host.all_modes()

    table_data = []
    for name, m in modes.items():
        apps_str = "\n".join(m.apps) if m.apps != ["*"] else "[dim]ALL PUBLIC[/dim]"
        up_str = "\n".join(m.bringup_apps) if m.bringup_apps else "[dim]None[/dim]"

        table_data.append([
            f"[green]{name}[/green]",
            apps_str,
            up_str,
            m.description
        ])

    print_simple_table(
        data=table_data,
        headers=["Name", "Apps (Allowed)", "Bring-up", "Description"],
        title="MOSS Discovered Modes",
        column_styles=["green", "cyan", "magenta", ""],
        title_style="bold yellow",
    )

    console.print(f"\n[dim]Total: {len(modes)} modes found.[/dim]")
    console.print(f"[dim]Use [bold]moss modes show <name>[/bold] to see instructions.[/dim]")
    _display_scan_errors(host.scan_errors)


@mode_app.command(name="show")
def show_mode(name: str):
    """
    Show detailed information and instructions for a specific mode.
    """
    host = Host()
    modes = host.all_modes()

    if name not in modes:
        console.print(f"[red]Error: Mode '{name}' not found.[/red]")
        raise typer.Exit(1)

    m = modes[name]

    # 基本信息
    content = (
        f"File Path: [dim]{m.file}[/dim]\n"
        f"Import Path: [dim]{m.import_path or 'N/A (Markdown Only)'}[/dim]\n"
        f"Description: [dim]{m.description}[/dim]"
    )
    print_simple_panel(content, title=f"Mode: {m.name}")

    # 指令内容
    if m.instruction:
        console.print("\n[bold cyan]Instruction (MODE.md):[/bold cyan]")
        console.print(Syntax(m.instruction, "markdown", theme="monokai", background_color="default"))
    else:
        console.print("\n[yellow]No custom instruction defined for this mode.[/yellow]")

    # Manifest 文件清单
    mode_dir = _resolve_mode_dir(m)
    if mode_dir:
        console.print("\n[bold cyan]Manifest files:[/bold cyan]")
        manifest_status = _list_manifest_files(mode_dir)
        # 根据 _ai_mode 选择输出方式
        from .utils import _ai_mode
        if _ai_mode:
            _print_manifest_files_ai(manifest_status)
        else:
            _print_manifest_files_human(manifest_status)
        console.print(f"\n[dim]Tip: [bold]moss --mode {name} manifests explain[/bold] for full capability view.[/dim]")

    _display_scan_errors(host.scan_errors)


def _resolve_mode_dir(mode) -> Path | None:
    """从 mode 对象解析其目录路径。"""
    # 优先从 file 字段推断
    if mode.file:
        p = Path(mode.file)
        if p.is_file():
            return p.parent
        if p.is_dir():
            return p
    # 从 import_path 推断
    if mode.import_path:
        import importlib
        try:
            mod = importlib.import_module(mode.import_path)
            if mod.__file__:
                return Path(mod.__file__).parent
        except ImportError:
            pass
    return None


@mode_app.command(name="create")
def create_mode(
        name: str = typer.Argument(..., help="Unique name for the new mode."),
        description: str = typer.Option("", "--desc", "-d", help="One-line description."),
        apps: List[str] = typer.Option([], "--app", "-a", help="Allowed app patterns (can repeat)."),
        up: List[str] = typer.Option([], "--up", "-u", help="Bring-up app patterns (can repeat)."),
):
    """
    Create a new MOSS Mode with a MODE.md file.
    """
    host = Host()
    try:
        mode_dir = host.new_mode(
            name=name,
            apps=apps,
            bringup_apps=up,
            description=description
        )
    except NameError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Failed to create mode:[/red] {e}")
        raise typer.Exit(1)

    console.print(f"[green]Mode '{name}' created.[/green]")

    # 列出创建的文件
    manifest_status = _list_manifest_files(mode_dir)
    from .utils import _ai_mode
    if _ai_mode:
        console.print("\nCreated files:")
        _print_manifest_files_ai(manifest_status)
        console.print(f"  MODE.md       (present)  — mode 元数据与指令")
        console.print(f"  __init__.py   (present)  — python package")
        console.print(f"\nNext:")
        console.print(f"  moss modes show {name}            — 查看 mode 详情")
        console.print(f"  moss --mode {name} manifests ...  — 在此 mode 下操作")
    else:
        console.print("\n[bold cyan]Created files:[/bold cyan]")
        _print_manifest_files_human(manifest_status)
        console.print("  [bold]MODE.md[/bold]       [green]present[/green]  [dim]— mode 元数据与指令[/dim]")
        console.print("  [bold]__init__.py[/bold]   [green]present[/green]  [dim]— python package[/dim]")
        console.print(f"\n[bold cyan]Next:[/bold cyan]")
        console.print(f"  [dim]moss modes show {name}[/dim]            — 查看 mode 详情")
        console.print(f"  [dim]moss --mode {name} manifests ...[/dim]  — 在此 mode 下操作")

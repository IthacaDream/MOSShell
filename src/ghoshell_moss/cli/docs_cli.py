"""
Docs CLI — browse and read MOSS reference documentation.

Default (no subcommand): show global intro with available doc sets.
Subcommands: list, read.

Doc sets: ai/ (--ai), en/ (--lang en, default), zh/ (--lang zh).
Each set can have its own README.md for set-level context.
"""

import subprocess
import typer
from pathlib import Path
from typing import Optional
from .utils import console, echo, print_error, print_info, print_simple_table, is_ai_mode

DOCS_ROOT = Path(__file__).resolve().parent / "docs"

docs_app = typer.Typer(
    name="docs",
    help="Browse and read MOSS reference documentation.",
    no_args_is_help=False,
    invoke_without_command=True,
)


# ---------------------------------------------------------------------------
# Root resolution
# ---------------------------------------------------------------------------

def _resolve_root(path: Optional[str], lang: str) -> Path:
    """Resolve the effective docs root from flags and mode."""
    if path:
        return Path(path).resolve()
    if is_ai_mode():
        return DOCS_ROOT / "ai"
    if lang and lang != "en":
        return DOCS_ROOT / lang
    return DOCS_ROOT


def _current_set_label(root: Path) -> str:
    """Human-readable label for the current doc set."""
    try:
        rel = root.relative_to(DOCS_ROOT)
        return str(rel) if str(rel) != "." else "root"
    except ValueError:
        return str(root)


# ---------------------------------------------------------------------------
# Available doc sets
# ---------------------------------------------------------------------------

def _available_doc_sets() -> list[dict]:
    """Discover available doc set subdirectories under DOCS_ROOT."""
    sets = []
    for d in sorted(DOCS_ROOT.iterdir()):
        if not d.is_dir() or d.name.startswith("."):
            continue
        label = {
            "ai": "AI documentation (use --ai)",
            "en": "English documentation (--lang en, default)",
            "zh": "Chinese documentation (--lang zh)",
        }.get(d.name, d.name)
        sets.append({"name": d.name, "label": label})
    return sets


def _print_doc_sets_hint(current: str = ""):
    """Print hint about available doc sets and how to switch."""
    sets = _available_doc_sets()
    if not sets:
        return
    echo("")
    echo("Doc sets:")
    for s in sets:
        marker = "  <-- current" if s["name"] == current else ""
        echo(f"  {s['name']}/  — {s['label']}{marker}")


# ---------------------------------------------------------------------------
# .gitignore helpers
# ---------------------------------------------------------------------------

def _read_gitignore(root: Path) -> list[str]:
    gitignore = root / ".gitignore"
    if not gitignore.exists():
        return []
    patterns = []
    with open(gitignore) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                patterns.append(line)
    return patterns


def _is_ignored(rel_path: str, patterns: list[str]) -> bool:
    import fnmatch
    for p in patterns:
        if fnmatch.fnmatch(rel_path, p) or fnmatch.fnmatch(rel_path, p.rstrip("/") + "/*"):
            return True
        if "/" in p or p.endswith("/"):
            if fnmatch.fnmatch(rel_path + "/", p) or fnmatch.fnmatch(rel_path, p.rstrip("/")):
                return True
    return False


# ---------------------------------------------------------------------------
# Heading / frontmatter extraction
# ---------------------------------------------------------------------------

def _first_heading(filepath: Path) -> str:
    try:
        with open(filepath) as f:
            for line in f:
                stripped = line.strip()
                if stripped.startswith("# ") and not stripped.startswith("## "):
                    return stripped[2:].strip()
    except Exception:
        pass
    return ""


def _frontmatter_title(filepath: Path) -> str | None:
    try:
        with open(filepath) as f:
            first = f.readline()
            if first.strip() != "---":
                return None
            for line in f:
                stripped = line.strip()
                if stripped == "---":
                    return None
                if stripped.startswith("title:") or stripped.startswith("title :"):
                    val = stripped.split(":", 1)[1].strip()
                    return val.strip('"').strip("'") or None
    except Exception:
        pass
    return None


def _doc_title(filepath: Path) -> str:
    return _frontmatter_title(filepath) or _first_heading(filepath) or filepath.stem


# ---------------------------------------------------------------------------
# README
# ---------------------------------------------------------------------------

def _read_readme(root: Path) -> str | None:
    """Read README.md content if it exists in the given root."""
    readme = root / "README.md"
    if readme.exists():
        return readme.read_text().strip()
    return None


# ---------------------------------------------------------------------------
# Git mtime
# ---------------------------------------------------------------------------

def _git_mtime(root: Path, rel_path: str) -> int:
    try:
        result = subprocess.run(
            ["git", "log", "-1", "--format=%ct", "--", rel_path],
            capture_output=True, text=True, cwd=root, timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            return int(result.stdout.strip())
    except Exception:
        pass
    try:
        return int((root / rel_path).stat().st_mtime)
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# Tree builder & output
# ---------------------------------------------------------------------------

def _build_tree(root: Path, patterns: list[str]) -> dict:
    tree: dict = {"dirs": {}, "files": []}

    for entry in sorted(root.iterdir(), key=lambda e: (not e.is_dir(), e.name)):
        rel = str(entry.relative_to(root))
        if _is_ignored(rel, patterns):
            continue
        if entry.name.startswith("."):
            continue
        if entry.is_dir():
            subtree = _build_tree(entry, patterns)
            if subtree["dirs"] or subtree["files"]:
                tree["dirs"][entry.name] = subtree
        elif entry.suffix == ".md":
            tree["files"].append({
                "path": rel,
                "name": entry.name,
                "heading": _first_heading(entry),
                "mtime": _git_mtime(root, rel),
            })

    tree["files"].sort(key=lambda f: f["mtime"], reverse=True)
    return tree


def _print_tree(root: Path, tree: dict):
    echo(str(root))
    for line in _tree_lines(tree):
        echo(line)


def _tree_lines(tree: dict, prefix: str = "") -> list[str]:
    lines = []
    dir_names = sorted(tree.get("dirs", {}).keys())
    files = tree.get("files", [])

    for i, name in enumerate(dir_names):
        is_last_dir = (i == len(dir_names) - 1) and not files
        connector = "└── " if is_last_dir else "├── "
        lines.append(f"{prefix}{connector}{name}/")
        sub_prefix = "    " if is_last_dir else "│   "
        lines.extend(_tree_lines(tree["dirs"][name], prefix + sub_prefix))

    for i, f in enumerate(files):
        connector = "└── " if i == len(files) - 1 else "├── "
        heading = f" — {f['heading']}" if f["heading"] else ""
        lines.append(f"{prefix}{connector}{f['name']}{heading}")

    return lines


# ---------------------------------------------------------------------------
# Collect all .md files under root (excluding README)
# ---------------------------------------------------------------------------

def _collect_docs(root: Path, patterns: list[str]) -> list[dict]:
    """Return list of {path, title} for all .md files under root, excluding README."""
    docs = []
    for entry in sorted(root.rglob("*.md"), key=lambda e: str(e)):
        rel = str(entry.relative_to(root))
        if _is_ignored(rel, patterns):
            continue
        if any(part.startswith(".") for part in entry.parts):
            continue
        if entry.name == "README.md":
            continue
        docs.append({
            "path": rel,
            "title": _doc_title(entry),
        })
    return docs


# ---------------------------------------------------------------------------
# Default callback — global intro or tree
# ---------------------------------------------------------------------------

@docs_app.callback(invoke_without_command=True)
def docs_callback(
        ctx: typer.Context,
        path: Optional[str] = typer.Option(
            None, "--path", "-p",
            help="Custom docs root path (bypasses --ai/--lang defaults).",
        ),
        lang: str = typer.Option(
            "en", "--lang", "-l",
            help="Language subdirectory for human docs (default: en).",
        ),
):
    """Browse and read MOSS reference documentation.

    \b
    moss docs                  show intro + available doc sets
    moss docs list             list docs in the current set
    moss docs read <path>      read a specific doc

    \b
    Switch doc sets:  --ai (ai/),  --lang en (en/),  --lang zh (zh/).
    """
    if ctx.invoked_subcommand is not None:
        ctx.obj = {"path": path, "lang": lang}
        return

    root = _resolve_root(path, lang)

    if is_ai_mode() and not path:
        # AI mode: compact tree view
        if not root.is_dir():
            print_error(f"Docs directory not found: {root}")
            raise typer.Exit(code=1)
        patterns = _read_gitignore(root)
        tree = _build_tree(root, patterns)
        if tree["dirs"] or tree["files"]:
            _print_tree(root, tree)
        else:
            print_info(f"No .md files found under {root}")
        _print_doc_sets_hint(current=_current_set_label(root))
        echo("")
        echo("Commands: moss docs list  |  moss docs read <path>")
        return

    # Human mode (or custom --path): global intro
    # Show root README if present
    readme = _read_readme(DOCS_ROOT)
    if readme:
        echo(readme)
    else:
        echo("MOSS Reference Documentation")

    echo("")

    # Show available doc sets
    sets = _available_doc_sets()
    if sets:
        echo("Available doc sets (switch with --ai or --lang):")
        echo("")
        for s in sets:
            echo(f"  {s['name']}/  — {s['label']}")
    else:
        echo("No doc sets found.")

    echo("")
    echo("Commands:")
    echo("  moss docs list              List docs in the current set")
    echo("  moss docs read <path>       Read a specific doc")
    echo("")
    echo("Examples:")
    echo("  moss --ai docs list          List AI-facing docs")
    echo("  moss docs --lang zh list     List Chinese docs")
    echo("  moss --ai docs read architecture-topology.md")


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------

@docs_app.command(name="list")
def list_docs(
        ctx: typer.Context,
        json_out: bool = typer.Option(False, "--json", help="JSON output for scripting."),
):
    """List docs in the current doc set, with hints for switching sets."""
    obj = ctx.obj or {}
    root = _resolve_root(obj.get("path"), obj.get("lang", "en"))
    current = _current_set_label(root)

    if not root.is_dir():
        print_error(f"Docs directory not found: {root}")
        raise typer.Exit(code=1)

    # Show set README as intro if present
    readme = _read_readme(root)
    if readme:
        echo(readme)
        echo("")

    patterns = _read_gitignore(root)
    docs = _collect_docs(root, patterns)

    if json_out:
        import json as j
        console.print(j.dumps(docs, ensure_ascii=False, indent=2))
        return

    if docs:
        rows = [[d["path"], d["title"]] for d in docs]
        print_simple_table(
            data=rows,
            headers=["Path", "Title"],
            title=f"Docs ({current}/) — {len(docs)} files",
        )
    else:
        print_info(f"No documents in {current}/ yet.")

    # Always show how to reach other doc sets
    _print_doc_sets_hint(current=current)
    echo("")
    echo("Read: moss docs read <path>")


# ---------------------------------------------------------------------------
# read
# ---------------------------------------------------------------------------

@docs_app.command(name="read")
def read_doc(
        ctx: typer.Context,
        doc_path: str = typer.Argument(help="Doc path relative to the current doc set."),
        raw: bool = typer.Option(False, "--raw", help="Output raw markdown without syntax highlighting."),
):
    """Read a reference documentation file by path."""
    obj = ctx.obj or {}
    root = _resolve_root(obj.get("path"), obj.get("lang", "en"))
    current = _current_set_label(root)

    if not root.is_dir():
        print_error(f"Docs directory not found: {root}")
        raise typer.Exit(code=1)

    filepath = root / doc_path
    if not filepath.exists() or not filepath.is_file():
        print_error(f"Document not found: {doc_path}")
        print_info(f"Current set: {current}/  (looked under: {root})")
        print_info("Use 'moss docs list' to see available documents.")
        raise typer.Exit(code=1)

    text = filepath.read_text()

    if raw or is_ai_mode():
        echo(text)
    else:
        from rich.syntax import Syntax
        echo(f"docs:{current}/{doc_path}\n")
        syntax = Syntax(text, "markdown", theme="monokai", line_numbers=True)
        console.print(syntax)

import typer

from ghoshell_moss.host import Host
from .utils import console, print_simple_table, print_simple_panel

ghost_app = typer.Typer(
    help="Manage MOSS Ghosts (Intelligent Agent Identities).",
    no_args_is_help=True,
)


@ghost_app.command(name="list")
def list_ghosts():
    """List all discovered ghosts in the current MOSS workspace."""
    host = Host()
    ghosts = host.all_ghosts()

    if not ghosts:
        console.print("[dim]No ghosts found in the workspace.[/dim]")
        console.print(
            "[dim]Place a GhostMeta instance in MOSS/ghosts/ "
            "(as a module or package) to make it discoverable.[/dim]"
        )
        return

    table_data = []
    for name, meta in ghosts.items():
        table_data.append([
            f"[green]{name}[/green]",
            meta.prototype(),
            meta.description().split("\n")[0][:120],
        ])

    print_simple_table(
        data=table_data,
        headers=["Name", "Prototype", "Description"],
        title="MOSS Discovered Ghosts",
    )
    console.print(f"\n[dim]Total: {len(ghosts)} ghost(s) found.[/dim]")
    console.print("[dim]Use [bold]moss ghosts show <name>[/bold] for details.[/dim]")


@ghost_app.command(name="show")
def show_ghost(name: str):
    """Show detailed information for a specific ghost."""
    host = Host()
    ghosts = host.all_ghosts()

    if name not in ghosts:
        console.print(f"[red]Error: Ghost '{name}' not found.[/red]")
        raise typer.Exit(1)

    meta = ghosts[name]

    # 基本信息
    print_simple_panel(
        f"Prototype:  [dim]{meta.prototype()}[/dim]\n"
        f"Version:    [dim]{meta.version() or '—'}[/dim]\n"
        f"Identifier: [dim]{meta.identifier}[/dim]\n"
        f"Description: [dim]{meta.description()}[/dim]",
        title=f"Ghost: {meta.name()}",
    )

    # Nuclei
    nuclei = meta.nuclei_metas()
    if nuclei:
        console.print("\n[bold cyan]Nuclei:[/bold cyan]")
        nuclei_data = [
            [f"[green]{n.name()}[/green]", n.description() or "—"]
            for n in nuclei
        ]
        print_simple_table(
            data=nuclei_data,
            headers=["Name", "Description"],
        )
    else:
        console.print("\n[dim]Nuclei: none declared[/dim]")

    # Providers
    providers = meta.providers()
    if providers:
        console.print("\n[bold cyan]Providers:[/bold cyan]")
        provider_data = [
            [f"[green]{p.__class__.__name__}[/green]", str(p.contract or "—")]
            for p in providers
        ]
        print_simple_table(
            data=provider_data,
            headers=["Provider Type", "Contract"],
        )
    else:
        console.print("\n[dim]Providers: none declared[/dim]")

    # Contracts
    contracts = meta.contracts()
    contract_list = list(contracts.contracts) if contracts else []
    if contract_list:
        console.print("\n[bold cyan]Contracts:[/bold cyan]")
        for c in contract_list:
            console.print(f"  [dim]•[/dim] {c}")
    else:
        console.print("\n[dim]Contracts: none declared[/dim]")

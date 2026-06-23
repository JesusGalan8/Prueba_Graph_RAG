"""
chat.py — Interfaz de chat interactiva en terminal usando Rich.

Uso:
  python -m src.cli.chat
  python -m src.cli.chat --verbose    # Muestra Cypher y debug info
  python -m src.cli.chat --no-color   # Sin colores (para terminales básicas)
"""
import argparse
import sys
import time
import uuid
from typing import Optional

from rich.console import Console
from rich.panel   import Panel
from rich.text    import Text
from rich.table   import Table
from rich.markdown import Markdown
from rich.spinner  import Spinner
from rich.live     import Live
from rich.rule     import Rule
from rich.style    import Style
from rich import box

from src.generation.generator     import F1Generator
from src.observability.tracing    import flush, is_enabled as langfuse_enabled
from src.config                   import OLLAMA_LLM_MODEL, SEASON_START, SEASON_END


# ─────────────────────────────────────────────────────────────
#  Estilos y constantes
# ─────────────────────────────────────────────────────────────

BANNER = """
[bold red]  ██████╗  [white]╔═══════════════════════════════════════╗
[bold red] ██╔════╝  [white]║  🏎️   F 1   G R A P H R A G          ║
[bold red] ██║  ███╗ [white]╠═══════════════════════════════════════╣
[bold red] ██║   ██║ [white]║  Fórmula 1 • Era Híbrida 2014–2024   ║
[bold red] ╚██████╔╝ [white]║  Neo4j + Llama 3.1 + Langfuse         ║
[bold red]  ╚═════╝  [white]╚═══════════════════════════════════════╝
"""

RETRIEVER_ICONS = {
    "CYPHER": "🔷",
    "VECTOR": "🔮",
    "HYBRID": "⚡",
}

RETRIEVER_COLORS = {
    "CYPHER": "bold cyan",
    "VECTOR": "bold magenta",
    "HYBRID": "bold yellow",
}

COMMANDS = {
    "/ayuda":   "Muestra esta ayuda",
    "/stats":   "Estadísticas del grafo",
    "/verbose": "Activa/desactiva modo detallado",
    "/limpiar": "Limpia la pantalla",
    "/salir":   "Sale del chat",
}


# ─────────────────────────────────────────────────────────────
#  Funciones de presentación
# ─────────────────────────────────────────────────────────────

def print_banner(console: Console) -> None:
    console.print(BANNER)
    info_parts = [
        f"[dim]LLM:[/dim] {OLLAMA_LLM_MODEL}",
        f"[dim]Datos:[/dim] {SEASON_START}–{SEASON_END}",
        f"[dim]Langfuse:[/dim] {'✓ activo' if langfuse_enabled() else '○ desactivado'}",
    ]
    console.print("  " + "  |  ".join(info_parts))
    console.print()
    console.print(Rule("[dim]Escribe tu pregunta o /ayuda para ver comandos[/dim]"))
    console.print()


def print_help(console: Console) -> None:
    table = Table(title="Comandos disponibles", box=box.ROUNDED, border_style="dim")
    table.add_column("Comando", style="bold cyan")
    table.add_column("Descripción")
    for cmd, desc in COMMANDS.items():
        table.add_row(cmd, desc)
    console.print(table)
    console.print()


def print_stats(console: Console, generator: F1Generator) -> None:
    """Muestra estadísticas del grafo directamente desde Neo4j."""
    import httpx
    try:
        r = httpx.get("http://localhost:8001/stats", timeout=5)
        data = r.json()
        table = Table(title="📊 Estadísticas del Grafo F1", box=box.ROUNDED)
        table.add_column("Entidad",  style="bold")
        table.add_column("Cantidad", justify="right", style="green")
        for k, v in data.items():
            table.add_row(k.capitalize(), f"{v:,}")
        console.print(table)
    except Exception:
        # Fallback: consulta directa a Neo4j
        console.print("[dim]Estadísticas no disponibles (API no conectada).[/dim]")
    console.print()


def print_response(console: Console, result: dict, verbose: bool) -> None:
    """Renderiza la respuesta del GraphRAG."""
    retriever    = result.get("retriever", "?")
    latency      = result.get("latency_ms", 0)
    answer       = result.get("answer", "")
    cypher       = result.get("cypher")
    trace_url    = result.get("trace_url")
    ret_latency  = result.get("ret_latency", 0)
    gen_latency  = result.get("gen_latency", 0)

    icon  = RETRIEVER_ICONS.get(retriever, "•")
    color = RETRIEVER_COLORS.get(retriever, "white")

    # Barra de estado
    status_parts = [
        f"[{color}]{icon} {retriever}[/{color}]",
        f"[dim]⏱ {latency:.0f}ms total[/dim]",
    ]
    if verbose:
        status_parts += [
            f"[dim]retrieval {ret_latency:.0f}ms[/dim]",
            f"[dim]generación {gen_latency:.0f}ms[/dim]",
        ]
    if trace_url:
        status_parts.append(f"[dim][link={trace_url}]📊 trace[/link][/dim]")

    console.print("  " + "  ".join(status_parts))

    # Cypher (si verbose)
    if verbose and cypher:
        console.print(Panel(
            f"[dim cyan]{cypher}[/dim cyan]",
            title="[dim]Cypher generado[/dim]",
            border_style="dim blue",
            padding=(0, 1),
        ))

    # Respuesta principal
    console.print(Panel(
        Markdown(answer),
        border_style="green",
        padding=(1, 2),
    ))
    console.print()


# ─────────────────────────────────────────────────────────────
#  Bucle principal del chat
# ─────────────────────────────────────────────────────────────

def run_chat(verbose: bool = False, no_color: bool = False) -> None:
    console = Console(highlight=not no_color)
    session_id = str(uuid.uuid4())[:8]

    print_banner(console)

    # Inicializar generador
    console.print("[dim]Inicializando conexiones...[/dim]")
    try:
        generator = F1Generator()
        console.print("[green]✓ Sistema listo.[/green]\n")
    except Exception as e:
        console.print(f"[bold red]✗ Error al inicializar: {e}[/bold red]")
        console.print("[dim]Asegúrate de que Neo4j y Ollama están corriendo.[/dim]")
        sys.exit(1)

    history = []

    try:
        while True:
            # Prompt de entrada
            try:
                console.print("[bold red]▶[/bold red] ", end="")
                question = input("").strip()
            except (KeyboardInterrupt, EOFError):
                break

            if not question:
                continue

            # ── Comandos especiales ──────────────────────
            if question.startswith("/"):
                cmd = question.lower()
                if cmd in ("/salir", "/exit", "/quit"):
                    break
                elif cmd == "/ayuda":
                    print_help(console)
                elif cmd == "/stats":
                    print_stats(console, generator)
                elif cmd == "/verbose":
                    verbose = not verbose
                    console.print(f"[dim]Modo verbose: {'ON' if verbose else 'OFF'}[/dim]\n")
                elif cmd == "/limpiar":
                    console.clear()
                    print_banner(console)
                elif cmd == "/historial":
                    for i, (q, a) in enumerate(history[-5:], 1):
                        console.print(f"[dim]{i}. Q: {q[:60]}...[/dim]")
                else:
                    console.print(f"[yellow]Comando desconocido: {question}[/yellow]")
                    console.print("[dim]Usa /ayuda para ver los comandos disponibles.[/dim]\n")
                continue

            # ── Consulta al GraphRAG ──────────────────────
            with Live(Spinner("dots", text="[dim]Consultando el grafo...[/dim]"),
                      refresh_per_second=10, console=console):
                result = generator.query(
                    question=question,
                    session_id=session_id,
                    verbose=verbose,
                )

            history.append((question, result.get("answer", "")))
            print_response(console, result, verbose)

    except Exception as e:
        console.print(f"\n[bold red]Error inesperado: {e}[/bold red]")
    finally:
        console.print(Rule())
        console.print("[dim]👋 ¡Hasta la próxima carrera! Cerrando conexiones...[/dim]")
        generator.close()
        flush()


# ─────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="F1 GraphRAG — Chat CLI")
    parser.add_argument("--verbose",  action="store_true",
                        help="Mostrar Cypher generado y tiempos detallados")
    parser.add_argument("--no-color", action="store_true",
                        help="Desactivar colores (terminales básicas)")
    args = parser.parse_args()
    run_chat(verbose=args.verbose, no_color=args.no_color)

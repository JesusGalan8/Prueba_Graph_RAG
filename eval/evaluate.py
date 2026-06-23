"""
evaluate.py — Script de evaluación del F1 GraphRAG contra el ground truth.

Evalúa en dos niveles:
  1. Nivel retrieval: ¿Los datos recuperados son correctos?
  2. Nivel generación: ¿La respuesta contiene la información correcta?

Uso:
  python eval/evaluate.py
  python eval/evaluate.py --output eval/results_2024.json
  python eval/evaluate.py --ids GT-001,GT-003  # Solo evaluar esas preguntas
  python eval/evaluate.py --quick              # Solo preguntas de dificultad baja
"""
import argparse
import json
import time
from pathlib import Path
from typing import Optional

# Añadir src al path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.table   import Table
from rich.panel   import Panel
from rich.rule    import Rule
from rich         import box

from src.generation.generator import F1Generator
from src.observability.tracing import flush


EVAL_DIR  = Path(__file__).parent
GT_FILE   = EVAL_DIR / "ground_truth.json"
OUT_FILE  = EVAL_DIR / "results.json"
console   = Console()


# ─────────────────────────────────────────────────────────────
#  Evaluación de una respuesta
# ─────────────────────────────────────────────────────────────

def evaluar_respuesta(respuesta: str, respuesta_esperada: str,
                       entidades: list[str]) -> dict:
    """
    Evalúa la respuesta en dos dimensiones:
    - exactitud: ¿Contiene los datos numéricos/nombres clave?
    - completitud: ¿Menciona todas las entidades esperadas?
    """
    resp_lower = respuesta.lower()
    exp_lower  = respuesta_esperada.lower()

    # Extraer números de la respuesta esperada y verificar si aparecen
    import re
    numeros_esperados = re.findall(r'\b\d+\b', exp_lower)
    nombres_esperados = [e.lower() for e in entidades if len(e) > 2]

    # Comprobar números clave
    numeros_encontrados = sum(
        1 for n in numeros_esperados
        if n in resp_lower
    )
    exactitud_numerica = (
        numeros_encontrados / len(numeros_esperados)
        if numeros_esperados else 1.0
    )

    # Comprobar entidades en la respuesta
    entidades_encontradas = sum(
        1 for e in nombres_esperados
        if e in resp_lower
    )
    cobertura_entidades = (
        entidades_encontradas / len(nombres_esperados)
        if nombres_esperados else 1.0
    )

    # Score combinado
    # Si acertó todos los números clave, la respuesta es correcta en la práctica
    if exactitud_numerica == 1.0:
        score = 1.0 if cobertura_entidades >= 0.5 else 0.8
    elif numeros_esperados:
        score = exactitud_numerica * 0.7 + cobertura_entidades * 0.3
    else:
        score = cobertura_entidades

    # Penalizar respuestas "no sé" cuando el ground truth tiene respuesta concreta
    no_data_phrases = ["no tengo", "no hay datos", "no puedo", "sin datos"]
    if any(p in resp_lower for p in no_data_phrases) and len(respuesta_esperada) > 5:
        score *= 0.1

    return {
        "score":                  round(score, 3),
        "exactitud_numerica":     round(exactitud_numerica, 3),
        "cobertura_entidades":    round(cobertura_entidades, 3),
        "numeros_esperados":      numeros_esperados,
        "numeros_encontrados":    numeros_encontrados,
        "entidades_evaluadas":    nombres_esperados,
        "entidades_encontradas":  entidades_encontradas,
    }


# ─────────────────────────────────────────────────────────────
#  Evaluación completa
# ─────────────────────────────────────────────────────────────

def run_evaluation(
    ids_filter:     Optional[list[str]] = None,
    quick:          bool = False,
    output_file:    Path = OUT_FILE,
) -> dict:
    """
    Ejecuta la evaluación completa contra el ground truth.

    Returns:
        dict con métricas globales y resultados por pregunta.
    """
    # Cargar ground truth
    with open(GT_FILE, encoding="utf-8") as f:
        ground_truth = json.load(f)

    # Filtros
    if ids_filter:
        ground_truth = [g for g in ground_truth if g["id"] in ids_filter]
    if quick:
        ground_truth = [g for g in ground_truth if g["dificultad"] == "baja"]

    console.print(Panel(
        f"[bold]Evaluando {len(ground_truth)} preguntas...[/bold]\n"
        f"Dataset: [cyan]{GT_FILE}[/cyan]",
        title="🏎️  F1 GraphRAG — Evaluación",
        border_style="blue",
    ))

    generator = F1Generator()
    resultados = []
    t0_global  = time.time()

    for i, gt in enumerate(ground_truth, 1):
        console.print(f"\n[dim]({i}/{len(ground_truth)})[/dim] [cyan]{gt['id']}[/cyan]: {gt['pregunta']}")

        t0 = time.time()
        try:
            result = generator.query(
                question=gt["pregunta"],
                session_id=f"eval-{gt['id']}",
            )
            respuesta  = result["answer"]
            retriever  = result["retriever"]
            latency    = result["latency_ms"]
            cypher     = result.get("cypher", "")
        except Exception as e:
            respuesta = f"ERROR: {e}"
            retriever = "ERROR"
            latency   = (time.time() - t0) * 1000
            cypher    = ""

        # Evaluar respuesta
        eval_result = evaluar_respuesta(
            respuesta=respuesta,
            respuesta_esperada=gt["respuesta_esperada"],
            entidades=gt.get("entidades", []),
        )

        score = eval_result["score"]
        color = "green" if score >= 0.7 else ("yellow" if score >= 0.4 else "red")
        console.print(
            f"  [{color}]Score: {score:.2f}[/{color}]  |  "
            f"Retriever: {retriever}  |  {latency:.0f}ms"
        )
        if score < 0.4:
            console.print(f"  [dim]Respuesta: {respuesta[:150]}...[/dim]")

        resultados.append({
            "id":                  gt["id"],
            "pregunta":            gt["pregunta"],
            "respuesta_esperada":  gt["respuesta_esperada"],
            "respuesta_obtenida":  respuesta,
            "retriever":           retriever,
            "cypher":              cypher,
            "latency_ms":          latency,
            "tipo":                gt.get("tipo", ""),
            "dificultad":          gt.get("dificultad", ""),
            "eval":                eval_result,
        })

    generator.close()
    flush()

    # ── Métricas globales ──────────────────────────────────
    total_time = time.time() - t0_global
    scores  = [r["eval"]["score"] for r in resultados]
    latencies = [r["latency_ms"] for r in resultados]

    metricas = {
        "total_preguntas":    len(resultados),
        "score_promedio":     round(sum(scores) / len(scores), 3) if scores else 0,
        "score_p75":          round(sorted(scores)[int(len(scores)*0.75)], 3) if scores else 0,
        "preguntas_ok":       sum(1 for s in scores if s >= 0.7),
        "preguntas_parcial":  sum(1 for s in scores if 0.4 <= s < 0.7),
        "preguntas_fail":     sum(1 for s in scores if s < 0.4),
        "latencia_promedio":  round(sum(latencies) / len(latencies), 0) if latencies else 0,
        "latencia_max":       round(max(latencies), 0) if latencies else 0,
        "tiempo_total_s":     round(total_time, 1),
        "por_tipo": {},
        "por_dificultad": {},
    }

    # Métricas por tipo
    tipos = set(r["tipo"] for r in resultados)
    for tipo in tipos:
        t_scores = [r["eval"]["score"] for r in resultados if r["tipo"] == tipo]
        metricas["por_tipo"][tipo] = round(sum(t_scores)/len(t_scores), 3) if t_scores else 0

    # Métricas por dificultad
    difs = set(r["dificultad"] for r in resultados)
    for dif in difs:
        d_scores = [r["eval"]["score"] for r in resultados if r["dificultad"] == dif]
        metricas["por_dificultad"][dif] = round(sum(d_scores)/len(d_scores), 3) if d_scores else 0

    # ── Guardar resultados ────────────────────────────────
    output = {"metricas": metricas, "resultados": resultados}
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    # ── Mostrar resumen ───────────────────────────────────
    _print_summary(metricas, output_file)
    return output


def _print_summary(metricas: dict, output_file: Path) -> None:
    score = metricas["score_promedio"]
    color = "green" if score >= 0.7 else ("yellow" if score >= 0.5 else "red")

    console.print()
    console.print(Rule("[bold]Resumen de Evaluación[/bold]"))

    # Tabla principal
    table = Table(box=box.ROUNDED, border_style="dim")
    table.add_column("Métrica",  style="bold")
    table.add_column("Valor",    justify="right")

    table.add_row("Score promedio",    f"[{color}]{metricas['score_promedio']:.1%}[/{color}]")
    table.add_row("Score P75",         f"{metricas['score_p75']:.1%}")
    table.add_row("✅ Correctas (≥0.7)", str(metricas['preguntas_ok']))
    table.add_row("⚠️  Parciales",       str(metricas['preguntas_parcial']))
    table.add_row("❌ Fallidas",         str(metricas['preguntas_fail']))
    table.add_row("Latencia promedio",  f"{metricas['latencia_promedio']:.0f}ms")
    table.add_row("Latencia máxima",    f"{metricas['latencia_max']:.0f}ms")
    table.add_row("Tiempo total",       f"{metricas['tiempo_total_s']:.1f}s")

    console.print(table)

    # Por dificultad
    table2 = Table(title="Score por dificultad", box=box.SIMPLE)
    table2.add_column("Dificultad")
    table2.add_column("Score", justify="right")
    for dif, sc in metricas["por_dificultad"].items():
        c = "green" if sc >= 0.7 else ("yellow" if sc >= 0.5 else "red")
        table2.add_row(dif, f"[{c}]{sc:.1%}[/{c}]")
    console.print(table2)

    console.print(f"\n[dim]Resultados guardados en: {output_file}[/dim]")


# ─────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="F1 GraphRAG — Evaluación")
    parser.add_argument("--ground-truth", type=Path, default=GT_FILE)
    parser.add_argument("--output",       type=Path, default=OUT_FILE)
    parser.add_argument("--ids",          type=str,  default=None,
                        help="IDs a evaluar, separados por coma (ej: GT-001,GT-003)")
    parser.add_argument("--quick",        action="store_true",
                        help="Solo preguntas de dificultad baja")
    args = parser.parse_args()

    ids_filter = [i.strip() for i in args.ids.split(",")] if args.ids else None

    run_evaluation(
        ids_filter=ids_filter,
        quick=args.quick,
        output_file=args.output,
    )

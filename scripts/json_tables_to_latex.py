from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any


TABLES_DIR = Path("tables")
TABLES_LATEX_DIR = Path("tables_latex")
STAT_COLUMN = "Estatística"
EXCLUDED_COLUMNS = {"cluster"}
RAW_STATS_PREFIX = "estatisticas_descritivas_"
TRANSFORMED_STATS_PREFIX = "dados_transformados_estatisticas_descritivas_"
STATS_OUTPUTS = {
    RAW_STATS_PREFIX: (
        "estatisticas_descritivas.tex",
        "Estatísticas descritivas dos dados originais",
    ),
    TRANSFORMED_STATS_PREFIX: (
        "dados_transformados_estatisticas_descritivas.tex",
        "Estatísticas descritivas dos dados transformados",
    ),
}
STAT_ORDER = [
    "Quantidade de Observações",
    "Quantidade de Valores Nulos",
    "Média Aritmética",
    "1º quartil",
    "Mediana",
    "3º quartil",
    "Desvio Padrão",
    "Amplitude",
    "Frequência Absoluta de Outliers",
    "Frequência Relativa de Outliers (%)",
]


LABEL_RE = re.compile(r"[^a-z0-9]+")


def escape_latex(value: Any) -> str:
    if value is None:
        return "--"

    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


def format_cell(value: Any) -> str:
    if value is None:
        return "--"
    if isinstance(value, bool):
        return "Sim" if value else "Não"
    if isinstance(value, int):
        return f"{value:,}".replace(",", r"\,")
    if isinstance(value, float):
        if math.isnan(value):
            return "--"
        if value.is_integer() and abs(value) >= 100:
            return f"{int(value):,}".replace(",", r"\,")
        if abs(value) >= 100:
            return f"{value:,.2f}".replace(",", r"\,")
        if abs(value) >= 1:
            return f"{value:.3f}"
        return f"{value:.4f}"
    return escape_latex(value)


def format_table_cell(value: Any, numeric_column: bool) -> str:
    if numeric_column and not isinstance(value, (int, float)) or isinstance(value, bool):
        return rf"\multicolumn{{1}}{{c}}{{{format_cell(value)}}}"
    return format_cell(value)


def bold(text: str) -> str:
    return rf"\textbf{{{text}}}"


def humanize(name: str) -> str:
    return name.replace("_", " ")


def caption_from_path(path: Path) -> str:
    stem = path.stem.replace("dados_transformados_", "dados transformados: ")
    return humanize(stem)


def label_from_path(path: Path) -> str:
    base = LABEL_RE.sub("-", path.stem.lower()).strip("-")
    return f"tab:{base}"


def ordered_columns(rows: list[dict[str, Any]]) -> list[str]:
    seen: list[str] = []
    for row in rows:
        for key in row:
            if key in EXCLUDED_COLUMNS:
                continue
            if key not in seen:
                seen.append(key)
    if STAT_COLUMN in seen:
        seen.remove(STAT_COLUMN)
        seen.insert(0, STAT_COLUMN)
    return seen


def single_stat_table(path: Path, row: dict[str, Any]) -> str:
    stat = row.get(STAT_COLUMN, caption_from_path(path))
    values = [(key, value) for key, value in row.items() if key != STAT_COLUMN]

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{{escape_latex(stat)}}}",
        rf"\label{{{label_from_path(path)}}}",
        r"\small",
        r"\setlength{\tabcolsep}{6pt}",
        r"\renewcommand{\arraystretch}{1.08}",
        r"\begin{tabular}{lr}",
        r"\toprule",
        rf"{bold('Variável')} & {bold('Valor')} \\",
        r"\midrule",
    ]
    lines.extend(
        rf"{bold(escape_latex(humanize(key)))} & {format_cell(value)} \\"
        for key, value in values
    )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def empty_table(path: Path) -> str:
    return "\n".join(
        [
            r"\begin{table}[htbp]",
            r"\centering",
            rf"\caption{{{escape_latex(caption_from_path(path))}}}",
            rf"\label{{{label_from_path(path)}}}",
            r"\small",
            r"\setlength{\tabcolsep}{6pt}",
            r"\renewcommand{\arraystretch}{1.08}",
            r"\begin{tabular}{l}",
            r"\toprule",
            rf"{bold('Resultado')} \\",
            r"\midrule",
            r"Nenhuma observação encontrada. \\",
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )


def regular_table(path: Path, rows: list[dict[str, Any]]) -> str:
    columns = ordered_columns(rows)
    alignment = "l" + "r" * (len(columns) - 1)

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{{escape_latex(caption_from_path(path))}}}",
        rf"\label{{{label_from_path(path)}}}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\renewcommand{\arraystretch}{1.08}",
        r"\resizebox{\textwidth}{!}{%",
        rf"\begin{{tabular}}{{{alignment}}}",
        r"\toprule",
        r"\multicolumn{1}{c}{"
        + bold(escape_latex(humanize(columns[0])))
        + r"} & "
        + " & ".join(
            rf"\multicolumn{{1}}{{c}}{{{bold(escape_latex(humanize(column)))}}}"
            for column in columns[1:]
        )
        + r" \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            format_cell(row.get(columns[0]))
            + " & "
            + " & ".join(
                format_table_cell(row.get(column), numeric_column=True)
                for column in columns[1:]
            )
            + r" \\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def stats_group_for_path(path: Path) -> str | None:
    name = path.name
    if name.startswith(TRANSFORMED_STATS_PREFIX):
        return TRANSFORMED_STATS_PREFIX
    if name.startswith(RAW_STATS_PREFIX):
        return RAW_STATS_PREFIX
    return None


def stat_sort_key(row: dict[str, Any]) -> tuple[int, str]:
    stat = str(row.get(STAT_COLUMN, ""))
    if stat in STAT_ORDER:
        return (STAT_ORDER.index(stat), stat)
    return (len(STAT_ORDER), stat)


def fused_stats_table(paths: list[Path], caption: str, output_name: str) -> str:
    rows: list[dict[str, Any]] = []
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list) or len(data) != 1 or not isinstance(data[0], dict):
            raise TypeError(f"{path}: expected one-row statistic table")
        rows.append(data[0])

    rows.sort(key=stat_sort_key)
    columns = ordered_columns(rows)
    alignment = "l" + "r" * (len(columns) - 1)

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{{escape_latex(caption)}}}",
        rf"\label{{{label_from_path(Path(output_name))}}}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\renewcommand{\arraystretch}{1.08}",
        r"\resizebox{\textwidth}{!}{%",
        rf"\begin{{tabular}}{{{alignment}}}",
        r"\toprule",
        r"\multicolumn{1}{c}{"
        + bold(escape_latex(humanize(columns[0])))
        + r"} & "
        + " & ".join(
            rf"\multicolumn{{1}}{{c}}{{{bold(escape_latex(humanize(column)))}}}"
            for column in columns[1:]
        )
        + r" \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            bold(format_cell(row.get(columns[0])))
            + " & "
            + " & ".join(
                format_table_cell(row.get(column), numeric_column=True)
                for column in columns[1:]
            )
            + r" \\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def latex_for_json(path: Path) -> str:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise TypeError(f"{path}: expected JSON array")
    if not rows:
        return empty_table(path)
    if not all(isinstance(row, dict) for row in rows):
        raise TypeError(f"{path}: expected array of objects")
    if len(rows) == 1 and STAT_COLUMN in rows[0]:
        return single_stat_table(path, rows[0])
    return regular_table(path, rows)


def main() -> None:
    TABLES_LATEX_DIR.mkdir(exist_ok=True)
    for tex_path in TABLES_LATEX_DIR.glob("*.tex"):
        tex_path.unlink()

    grouped_stats: dict[str, list[Path]] = {prefix: [] for prefix in STATS_OUTPUTS}
    for json_path in sorted(TABLES_DIR.glob("*.json")):
        group = stats_group_for_path(json_path)
        if group is not None:
            grouped_stats[group].append(json_path)
            continue

        tex_path = TABLES_LATEX_DIR / json_path.with_suffix(".tex").name
        tex_path.write_text(latex_for_json(json_path), encoding="utf-8")

    for prefix, paths in grouped_stats.items():
        output_name, caption = STATS_OUTPUTS[prefix]
        tex_path = TABLES_LATEX_DIR / output_name
        tex_path.write_text(
            fused_stats_table(paths, caption, output_name),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()

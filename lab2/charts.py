"""直接输出可缩放 SVG 图表，无第三方依赖；条形刻度在各场景内独立。"""
from html import escape
from pathlib import Path

from .fixtures import pdf_samples

PALETTE = ("#53799d", "#178575", "#df8844", "#8766b2")


def _text(x, y, text, *, size=13, color="#263445", anchor="start"):
    return (f'<text x="{x}" y="{y}" fill="{color}" font-size="{size}" '
            f'text-anchor="{anchor}" font-family="Arial, sans-serif">{escape(str(text))}</text>')


def _save(path: Path, elements: list[str], width: int, height: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
                    f'viewBox="0 0 {width} {height}">\n<rect width="100%" height="100%" fill="#fafbfd"/>\n'
                    + "\n".join(elements) + "\n</svg>\n", encoding="utf-8")


def panel_bars(path: Path, title: str, panels: list[tuple[str, list[tuple[str, float, float | None, float | None]]]],
               unit: str):
    width, left, plot_width = 1000, 265, 605
    elements = [_text(28, 35, title, size=22),
                _text(28, 62, f"Independent scale per scenario. {unit}; timing whiskers = 25th-75th percentile.", size=13)]
    y = 100
    for scenario, values in panels:
        limit = max((hi if hi is not None else value for _, value, _, hi in values), default=1) or 1
        elements.append(_text(28, y, scenario, size=16))
        elements.append(_text(left + plot_width, y, f"panel max: {limit:,.1f} {unit}", size=11, anchor="end"))
        y += 18
        for i, (label, value, lo, hi) in enumerate(values):
            elements.append(_text(left - 12, y + 16, label, size=12, anchor="end"))
            elements.append(f'<rect x="{left}" y="{y}" width="{plot_width}" height="22" fill="#e8edf2"/>')
            bar = plot_width * value / limit
            elements.append(f'<rect x="{left}" y="{y}" width="{bar:.3f}" height="22" '
                            f'fill="{PALETTE[i % len(PALETTE)]}"><title>{escape(label)}: {value}</title></rect>')
            if lo is not None and hi is not None:
                a, b, mid = left + plot_width * lo / limit, left + plot_width * hi / limit, y + 11
                elements.append(f'<path d="M {a:.2f},{mid} H {b:.2f} M {a:.2f},{mid-5} V {mid+5} '
                                f'M {b:.2f},{mid-5} V {mid+5}" stroke="#172536" stroke-width="1.5"/>')
            elements.append(_text(left + plot_width + 12, y + 16,
                                  f"{value:,.2f}" if unit == "us" else f"{value:,.0f}", size=12))
            y += 30
        y += 32
    _save(path, elements, width, y + 5)


def matrix_chart(path: Path, labels, matrix):
    elements = [_text(28, 34, "Treasure: A* minimum entry costs", size=21)]
    left, top, cell = 100, 90, 70
    maximum = max(max(r) for r in matrix) or 1
    for i, label in enumerate(labels):
        elements.append(_text(left + i * cell + cell / 2, top - 16, label, anchor="middle"))
        elements.append(_text(left - 14, top + i * cell + 42, label, anchor="end"))
        for j, value in enumerate(matrix[i]):
            shade = int(245 - 125 * value / maximum)
            color = f"rgb({shade},{min(255, shade + 12)},{min(255, shade + 20)})"
            x, y = left + j * cell, top + i * cell
            elements.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="{color}" '
                            'stroke="#ffffff" stroke-width="2"/>')
            elements.append(_text(x + cell / 2, y + 42, value, size=19, anchor="middle"))
    _save(path, elements, 600, 540)


def sample_paths_chart(path: Path, standards: dict):
    data = pdf_samples()
    elements = [_text(28, 33, "PDF samples: computed shortest paths", size=22),
                _text(28, 58, "S is not charged. Maze: 7 moves / 8 coordinates. Weighted: E entry cost = 0.")]
    cell, left = 56, 50
    for index, key in enumerate(("maze", "weighted")):
        top = 120 + index * 380
        cells = data[key]["cells"]
        result = standards[key][-1]
        path_coords = [tuple(p) for p in result["path"]]
        elements.append(_text(left, top - 25, f"{key}: cost = {result['cost']}", size=17))
        for r, row in enumerate(cells):
            for c, label in enumerate(row):
                blocked = label == "X"
                color = "#465264" if blocked else "#cff0e5" if (r, c) in path_coords else "#edf1f5"
                x, y = left + c * cell, top + r * cell
                elements.append(f'<rect x="{x}" y="{y}" width="{cell-2}" height="{cell-2}" fill="{color}"/>')
                elements.append(_text(x + cell / 2, y + 23, label, size=15,
                                      color="white" if blocked else "#263445", anchor="middle"))
                if (r, c) in path_coords:
                    elements.append(_text(x + cell / 2, y + 43, f"step {path_coords.index((r,c))}",
                                          size=10, anchor="middle"))
        right = left + 5 * cell + 40
        elements.append(_text(right, top + 15, "Cumulative cost (g)", size=15))
        g = 0
        for step, (r, c) in enumerate(path_coords):
            entry = 0 if not step or (key == "weighted" and cells[r][c] == "E") else (1 if key == "maze" else int(cells[r][c]))
            g += entry
            elements.append(_text(right, top + 45 + step * 24, f"{step}: ({r},{c})  +{entry}  =>  {g}", size=12))
    _save(path, elements, 710, 790)


def generate_charts(summary: list[dict], directory: Path, standard: dict, matrix: dict):
    paths = list(dict.fromkeys(r["scenario"] for r in summary if r["family"] == "path"))
    games = list(dict.fromkeys(r["scenario"] for r in summary if r["family"] in {"tree", "treasure"}))

    def selected(scenario, **fields):
        return next(r for r in summary if r["scenario"] == scenario and all(r[k] == v for k, v in fields.items()))

    def count_value(label, row, field):
        return label, row[field], None, None

    def time_value(label, row):
        return (label, row["total_ns_median"] / 1000,
                row["total_ns_p25"] / 1000, row["total_ns_p75"] / 1000)

    for filename, title, selector, timing in (
        ("astar_heuristics_nodes.svg", "A*: heuristic comparison (FIFO)", "heuristic", False),
        ("astar_tie_break_nodes.svg", "A*: tie breaking (admissible Manhattan)", "tie", False),
        ("astar_total_time.svg", "A*: total time including heuristic preprocessing (larger g first)", "heuristic", True)):
        panels = []
        for scenario in paths:
            values = []
            labels = ("zero", "manhattan", "relaxed") if selector == "heuristic" else ("fifo", "larger_g", "smaller_g")
            for label in labels:
                row = selected(scenario, heuristic=label if selector == "heuristic" else "manhattan",
                               tie_break=("larger_g" if timing else "fifo") if selector == "heuristic" else label)
                values.append(time_value(label, row) if timing else count_value(label, row, "expanded_nodes"))
            panels.append((scenario, values))
        panel_bars(directory / filename, title, panels, "us" if timing else "nodes")
    for filename, field, title, timing in (
        ("game_recursive_visits.svg", "recursive_visits", "Game search: recursion visits", False),
        ("game_leaf_evaluations.svg", "leaf_evaluations", "Game search: leaf evaluations", False),
        ("game_total_time.svg", "", "Game search: total time including node ordering", True)):
        panels = []
        for scenario in games:
            values = []
            for algorithm, order in (("minimax", "original"), ("alphabeta", "original"),
                                     ("alphabeta", "reverse"), ("alphabeta", "heuristic")):
                row = selected(scenario, algorithm=algorithm, order=order)
                label = ("MM" if algorithm == "minimax" else "AB") + "/" + order
                values.append(time_value(label, row) if timing else count_value(label, row, field))
            panels.append((scenario, values))
        panel_bars(directory / filename, title, panels, "us" if timing else "nodes")
    matrix_chart(directory / "distance_matrix.svg", matrix["labels"], matrix["matrix"])
    sample_paths_chart(directory / "sample_paths.svg", standard)
    files = sorted(p.name for p in directory.glob("*.svg"))
    html = ('<!doctype html><html lang="en"><meta charset="utf-8"><title>Lab 2 charts</title>'
            '<style>body{max-width:1040px;margin:30px auto;background:#f0f3f7;font:16px Arial}'
            'img{width:100%;height:auto;background:white;margin:15px 0}a{color:#157665}</style>'
            '<h1>Lab 2 reproducible experiment charts</h1><p>Source: ../summary.csv. '
            'Each scenario panel uses its own scale. Timing bars show medians and interquartile ranges.</p>')
    html += "".join(f'<p><a href="{escape(name)}">{escape(name)}</a></p><img src="{escape(name)}" alt="{escape(name)}">'
                    for name in files)
    (directory / "index.html").write_text(html + "</html>", encoding="utf-8")

"""一致输入、固定种子、随机交错配置、批量重复计时，保存每次测量。"""
import math
import platform
import random
import statistics
import sys
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from time import get_clock_info, perf_counter_ns

from .fixtures import path_scenarios, pdf_samples, pdf_tree, tree_scenarios, tree_to_dict
from .game_tree import alphabeta, minimax
from .io_utils import write_csv, write_json
from .pathfinding import astar, build_heuristic, maze_grid, weighted_grid
from .treasure import GameState, TreasureGame, distance_matrix, treasure_alphabeta, treasure_minimax


def pdf_game():
    sample = pdf_samples()["treasure"]
    grid = weighted_grid(sample["cells"], end_cost=1)
    matrix, paths = distance_matrix(grid, tuple(map(tuple, sample["positions"])))
    game = TreasureGame(tuple(sample["labels"]),
                        tuple(sample["values"].get(label, 0) for label in sample["labels"]), matrix)
    return game, paths


def export_standard_results(directory: Path) -> None:
    data = pdf_samples()
    samples = {}
    steps = []
    for key, builder in (("maze", maze_grid), ("weighted", weighted_grid)):
        sample = data[key]
        grid, start, goal = builder(sample["cells"]), tuple(sample["start"]), tuple(sample["goal"])
        alternatives = []
        for mode in ("zero", "manhattan"):
            h = build_heuristic(grid, goal, mode)
            result = astar(grid, start, goal, heuristic=h)
            expected = sample["expected_steps" if key == "maze" else "expected_cost"]
            if result.cost != expected:
                raise AssertionError(f"PDF {key} 验证失败")
            alternatives.append({"heuristic": mode, "path": result.path, "cost": result.cost,
                                 "steps": len(result.path) - 1, "metrics": asdict(result.metrics)})
        samples[key] = alternatives
        cumulative = 0
        h = build_heuristic(grid, goal)
        for i, p in enumerate(alternatives[-1]["path"]):
            entry = grid.cost(p) if i else 0
            cumulative += entry
            steps.append({"problem": key, "step": i, "row": p[0], "col": p[1],
                          "entry_cost": entry, "g": cumulative, "h": h(p), "f": cumulative + h(p)})
    tree_results = {}
    for name, solver in (("minimax", minimax), ("alphabeta", alphabeta)):
        result = solver(pdf_tree(), trace=True)
        if result.value != data["tree"]["expected_value"]:
            raise AssertionError("PDF 博弈树验证失败")
        tree_results[name] = {"value": result.value, "metrics": asdict(result.metrics), "trace": result.trace}
    samples["tree"] = tree_results
    game, paths = pdf_game()
    write_json(directory / "distance_matrix.json", {"labels": game.labels, "matrix": game.distances})
    write_json(directory / "distance_paths.json", paths)
    decisions, action_values, trajectories = [], [], []
    for depth in (2, 4):
        exact = {}
        for t in GameState().remaining:
            child = game.apply(GameState(), t)
            mm = treasure_minimax(game, depth - 1, state=child)
            ab = treasure_alphabeta(game, depth - 1, state=child)
            if mm.best_value != ab.best_value:
                raise AssertionError("全窗口首步分支评价不一致")
            exact[game.labels[t]] = mm.best_value
            action_values.append({"depth": depth, "target": game.labels[t],
                                  "a_move_cost": game.distances[0][t], "a_gain": game.gain(GameState(), t),
                                  "minimax_value": mm.best_value, "alphabeta_full_window_value": ab.best_value})
        for name, solver in (("minimax", treasure_minimax), ("alphabeta", treasure_alphabeta)):
            result = solver(game, depth, trace=True)
            expected = data["treasure"]["expected"][str(depth)]
            if (result.best_target, result.best_value) != (expected["best_target"], expected["best_value"]):
                raise AssertionError(f"PDF 宝藏 H={depth} 验证失败")
            decisions.append({"depth": depth, "algorithm": name, "order": "original",
                              "best_target": result.best_target, "best_value": result.best_value,
                              "metrics": asdict(result.metrics), "trace": result.trace})
        state = GameState()
        for step in range(depth):
            result = treasure_minimax(game, depth - step, state=state)
            t = game.labels.index(result.best_target)
            after = game.apply(state, t)
            trajectories.append({"depth": depth, "step": step + 1, "player": state.player,
                                 "from": game.labels[state.pos_a if state.player == "A" else state.pos_b],
                                 "target": game.labels[t], "gain": game.gain(state, t),
                                 "score_a": after.score_a, "score_b": after.score_b, "utility": after.utility})
            state = after
    samples["treasure"] = decisions
    write_json(directory / "standard_results.json", samples)
    write_csv(directory / "path_cost_steps.csv", steps)
    write_csv(directory / "root_action_values.csv", action_values)
    write_csv(directory / "optimal_trajectories.csv", trajectories)


def _quantile(values: list[float], fraction: float) -> float:
    values = sorted(values)
    pos = (len(values) - 1) * fraction
    lo, hi = math.floor(pos), math.ceil(pos)
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)


def _validate_config(config: dict) -> None:
    for name in ("repeats", "warmups", "max_batch_iterations"):
        if type(config[name]) is not int or config[name] < 1:
            raise ValueError(f"{name} 必须是正整数")
    if config["repeats"] < 5 or config["target_batch_ms"] <= 0:
        raise ValueError("性能测量至少重复 5 次，批量时间必须为正")
    if set(config["heuristics"]) != {"zero", "manhattan", "relaxed"} or len(config["heuristics"]) != 3:
        raise ValueError("必须保留全部三种启发函数对照")
    if set(config["tie_breaks"]) != {"fifo", "larger_g", "smaller_g"} or len(config["tie_breaks"]) != 3:
        raise ValueError("必须保留全部三种队列策略")
    if set(config["orders"]) != {"original", "reverse", "heuristic"} or len(config["orders"]) != 3:
        raise ValueError("必须保留全部三种访问顺序")
    if config["game_depths"] != [2, 4]:
        raise ValueError("综合题正式对照深度应为 [2,4]")


def run_experiments(config: dict, directory: Path) -> list[dict]:
    _validate_config(config)
    seed = config["seed"]
    paths, trees = path_scenarios(seed), tree_scenarios(seed)
    game, _ = pdf_game()
    configurations = []
    inputs = {"path_scenarios": [], "tree_scenarios": []}

    def register(meta: dict, function, expected, optimal_targets=None):
        configurations.append({"meta": {"config_id": f"c{len(configurations):03d}", **meta},
                               "call": function, "expected": expected, "optimal_targets": optimal_targets})

    for name, grid, start, goal in paths:
        inputs["path_scenarios"].append({"name": name, "costs": grid.costs, "start": start, "goal": goal})
        expected = astar(grid, start, goal, heuristic=build_heuristic(grid, goal, "zero")).cost
        for mode in config["heuristics"]:
            for tie in config["tie_breaks"]:
                def run(grid=grid, start=start, goal=goal, mode=mode, tie=tie):
                    before = perf_counter_ns()
                    h = build_heuristic(grid, goal, mode)
                    prepare = perf_counter_ns() - before
                    result = astar(grid, start, goal, heuristic=h, tie_break=tie)
                    return result, prepare
                register({"family": "path", "scenario": name, "algorithm": "astar",
                          "heuristic": mode, "tie_break": tie, "order": "", "depth": ""}, run, expected)
    for name, tree in trees:
        inputs["tree_scenarios"].append({"name": name, "tree": tree_to_dict(tree)})
        expected = minimax(tree).value
        for algorithm, solver in (("minimax", minimax), ("alphabeta", alphabeta)):
            for order in config["orders"]:
                register({"family": "tree", "scenario": name, "algorithm": algorithm,
                          "heuristic": "", "tie_break": "", "order": order, "depth": ""},
                         lambda solver=solver, tree=tree, order=order: (solver(tree, order=order), 0), expected)
    for depth in config["game_depths"]:
        exact = {game.labels[t]: treasure_minimax(game, depth - 1, state=game.apply(GameState(), t)).best_value
                 for t in GameState().remaining}
        expected = max(exact.values())
        optimal = {t for t, v in exact.items() if v == expected}
        for algorithm, solver in (("minimax", treasure_minimax), ("alphabeta", treasure_alphabeta)):
            for order in config["orders"]:
                register({"family": "treasure", "scenario": f"treasure_H{depth}", "algorithm": algorithm,
                          "heuristic": "", "tie_break": "", "order": order, "depth": depth},
                         lambda solver=solver, depth=depth, order=order: (solver(game, depth, order=order), 0),
                         expected, optimal)
    write_json(directory / "scenario_inputs.json", inputs)

    def validate(item, result):
        value = result.cost if item["meta"]["family"] == "path" else (
            result.best_value if item["meta"]["family"] == "treasure" else result.value)
        if value != item["expected"]:
            raise AssertionError(f"{item['meta']} 得分/最优代价改变")
        if item["optimal_targets"] is not None and result.best_target not in item["optimal_targets"]:
            raise AssertionError("排序后首步不属于真正最优行动集合")
        return value

    # 预热和批量大小校准；这些测量不作为正式数据。
    for item in configurations:
        for _ in range(config["warmups"]):
            result, _ = item["call"]()
            validate(item, result)
        start = perf_counter_ns()
        result, _ = item["call"]()
        duration = max(1, perf_counter_ns() - start)
        item["batch"] = max(1, min(config["max_batch_iterations"],
                                   math.ceil(config["target_batch_ms"] * 1_000_000 / duration)))
        item["counts"] = result.metrics.counts()
    rows = []
    for trial in range(config["repeats"]):
        shuffled = configurations.copy()
        random.Random(seed + trial).shuffle(shuffled)
        for position, item in enumerate(shuffled):
            prepare_sum = search_sum = 0
            before = perf_counter_ns()
            for _ in range(item["batch"]):
                result, prepare = item["call"]()
                prepare_sum += prepare
                search_sum += result.metrics.elapsed_ns
            elapsed = perf_counter_ns() - before
            value = validate(item, result)
            if result.metrics.counts() != item["counts"]:
                raise AssertionError("同一配置的计数不可复现")
            rows.append({**item["meta"], "trial": trial, "execution_order": position,
                         "iterations": item["batch"], "result_value": value,
                         "best_target": getattr(result, "best_target", ""),
                         **result.metrics.counts(), "preprocess_ns": prepare_sum / item["batch"],
                         "search_ns": search_sum / item["batch"], "total_ns": elapsed / item["batch"]})
        print(f"性能测量 {trial + 1}/{config['repeats']} 完成", flush=True)
    summary = []
    for item in configurations:
        samples = [row for row in rows if row["config_id"] == item["meta"]["config_id"]]
        row = {**item["meta"], "samples": len(samples), "iterations": item["batch"],
               "result_value": samples[0]["result_value"], "best_target": samples[0]["best_target"], **item["counts"]}
        for field in ("preprocess_ns", "search_ns", "total_ns"):
            values = [s[field] for s in samples]
            for suffix, value in (("median", statistics.median(values)), ("mean", statistics.mean(values)),
                                  ("stdev", statistics.stdev(values)), ("min", min(values)), ("max", max(values)),
                                  ("p25", _quantile(values, .25)), ("p75", _quantile(values, .75))):
                row[f"{field}_{suffix}"] = value
        summary.append(row)
    write_csv(directory / "raw_measurements.csv", rows)
    write_csv(directory / "summary.csv", summary)
    export_comparisons(summary, directory)
    clock = get_clock_info("perf_counter")
    write_json(directory / "environment.json", {
        "timestamp_asia_shanghai": datetime.now(timezone(timedelta(hours=8))).isoformat(),
        "python": sys.version, "platform": platform.platform(),
        "processor": platform.processor(), "machine": platform.machine(),
        "clock": {"implementation": clock.implementation, "resolution_seconds": clock.resolution,
                  "monotonic": clock.monotonic}, "config": config,
        "method": {"input_construction_timed": False, "distance_precompute_timed_in_game": False,
                   "heuristic_preprocessing_in_total": True, "ordering_in_search_time": True,
                   "gc": "Python 默认开启；所有配置一致", "shuffle_seed": "seed+trial",
                   "nanoseconds": "每个批量重复的平均单次耗时；总耗时包含调用和记录计时的开销"}
    })
    return summary


def export_comparisons(summary: list[dict], directory: Path) -> None:
    """只导出可从原始汇总重算的差值/比例，不生成报告文字。"""
    comparisons = []

    def compare(kind: str, baseline: dict, variant: dict, field: str):
        base_nodes, other_nodes = baseline[field], variant[field]
        comparisons.append({
            "comparison": kind, "scenario": baseline["scenario"],
            "baseline_config": baseline["config_id"], "variant_config": variant["config_id"],
            "count_metric": field, "baseline_count": base_nodes, "variant_count": other_nodes,
            "count_reduction_percent": 100 * (base_nodes - other_nodes) / base_nodes if base_nodes else 0,
            "baseline_total_us_median": baseline["total_ns_median"] / 1000,
            "variant_total_us_median": variant["total_ns_median"] / 1000,
            "median_speedup": baseline["total_ns_median"] / variant["total_ns_median"],
            "baseline_preprocess_us_median": baseline["preprocess_ns_median"] / 1000,
            "variant_preprocess_us_median": variant["preprocess_ns_median"] / 1000,
            "result_value": baseline["result_value"],
            "same_result": baseline["result_value"] == variant["result_value"]})

    for scenario in dict.fromkeys(r["scenario"] for r in summary):
        rows = [r for r in summary if r["scenario"] == scenario]
        if rows[0]["family"] == "path":
            lookup = {(r["heuristic"], r["tie_break"]): r for r in rows}
            for tie in ("fifo", "larger_g", "smaller_g"):
                compare(f"zero_vs_manhattan/{tie}", lookup["zero", tie], lookup["manhattan", tie], "expanded_nodes")
                compare(f"manhattan_vs_relaxed/{tie}", lookup["manhattan", tie], lookup["relaxed", tie], "expanded_nodes")
            for tie in ("larger_g", "smaller_g"):
                compare(f"fifo_vs_{tie}/manhattan", lookup["manhattan", "fifo"], lookup["manhattan", tie], "expanded_nodes")
        else:
            lookup = {(r["algorithm"], r["order"]): r for r in rows}
            compare("minimax_vs_alphabeta/original", lookup["minimax", "original"], lookup["alphabeta", "original"], "recursive_visits")
            for order in ("reverse", "heuristic"):
                compare(f"original_vs_{order}/alphabeta", lookup["alphabeta", "original"], lookup["alphabeta", order], "recursive_visits")
    write_csv(directory / "effect_comparisons.csv", comparisons)

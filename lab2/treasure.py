"""位置、剩余集合、双边收益、当前玩家组成完整不可变博弈状态。"""
from dataclasses import dataclass
from math import inf
from time import perf_counter_ns

from .metrics import Metrics
from .pathfinding import Coord, Grid, PathResult, astar, build_heuristic


def distance_matrix(grid: Grid, positions: tuple[Coord, ...]) -> tuple[tuple[tuple[int, ...], ...], list[dict]]:
    """逐个有向位置对调用 A*；不假定进入代价距离天然对称。"""
    matrix, records = [], []
    for i, start in enumerate(positions):
        row = []
        for j, goal in enumerate(positions):
            result: PathResult = astar(grid, start, goal, heuristic=build_heuristic(grid, goal),
                                       tie_break="larger_g")
            row.append(result.cost)
            records.append({"from_index": i, "to_index": j, "cost": result.cost,
                            "path": result.path, "metrics": result.metrics.counts()})
        matrix.append(tuple(row))
    return tuple(matrix), records


@dataclass(frozen=True)
class GameState:
    pos_a: int = 0
    pos_b: int = 1
    remaining: tuple[int, ...] = (2, 3, 4, 5)
    score_a: int = 0
    score_b: int = 0
    player: str = "A"

    @property
    def utility(self) -> int:
        return self.score_a - self.score_b


@dataclass(frozen=True)
class TreasureGame:
    labels: tuple[str, ...]
    values: tuple[int, ...]
    distances: tuple[tuple[int, ...], ...]

    def __post_init__(self):
        n = len(self.labels)
        if n < 2 or len(set(self.labels)) != n or len(self.values) != n or len(self.distances) != n:
            raise ValueError("标签、价值及矩阵维度不匹配")
        if any(type(v) is not int for v in self.values):
            raise ValueError("价值必须为整数")
        if any(len(row) != n for row in self.distances):
            raise ValueError("距离矩阵必须为方阵")
        if any(type(d) is not int or d < 0 for row in self.distances for d in row):
            raise ValueError("本博弈要求关键位置相互可达；不可达距离不能当成负移动代价")
        if any(self.distances[i][i] != 0 for i in range(n)):
            raise ValueError("矩阵对角线必须为零")

    def validate(self, state: GameState) -> None:
        n = len(self.labels)
        if state.player not in {"A", "B"} or any(type(x) is not int or not 0 <= x < n
                                                for x in (state.pos_a, state.pos_b)):
            raise ValueError("无效玩家或位置")
        if len(set(state.remaining)) != len(state.remaining) or any(
                type(x) is not int or not 2 <= x < n for x in state.remaining):
            raise ValueError("剩余宝藏必须互异且对应宝藏编号")
        if type(state.score_a) is not int or type(state.score_b) is not int:
            raise ValueError("累计收益必须为整数")

    def gain(self, state: GameState, target: int) -> int:
        pos = state.pos_a if state.player == "A" else state.pos_b
        return self.values[target] - self.distances[pos][target]

    def apply(self, state: GameState, target: int) -> GameState:
        if target not in state.remaining:
            raise ValueError("只能选择尚未取得的宝藏")
        gain = self.gain(state, target)
        remaining = tuple(x for x in state.remaining if x != target)
        if state.player == "A":
            return GameState(target, state.pos_b, remaining, state.score_a + gain, state.score_b, "B")
        return GameState(state.pos_a, target, remaining, state.score_a, state.score_b + gain, "A")

    def actions(self, state: GameState, order: str, stats: Metrics) -> list[int]:
        if order == "original":
            return list(state.remaining)
        if order == "reverse":
            return list(reversed(state.remaining))
        if order != "heuristic":
            raise ValueError("顺序必须为 original/reverse/heuristic")
        stats.ordering_probes += len(state.remaining)
        # A/B 都优先自己的即时收益更高的宝藏。Python 稳定排序保留平局顺序。
        return sorted(state.remaining, key=lambda t: self.gain(state, t), reverse=True)


@dataclass
class GameResult:
    best_target: str | None
    best_value: int
    metrics: Metrics
    trace: list[dict]


def _check(game: TreasureGame, state: GameState, depth: int, order: str) -> None:
    game.validate(state)
    if type(depth) is not int or depth < 0:
        raise ValueError("H 必须为非负整数，单位为单方行动次数")
    if order not in {"original", "reverse", "heuristic"}:
        raise ValueError("无效行动顺序")


def treasure_minimax(game: TreasureGame, depth: int, *, state: GameState = GameState(),
                     order: str = "original", trace: bool = False) -> GameResult:
    _check(game, state, depth, order)
    stats, events = Metrics(), []
    began = perf_counter_ns()

    def visit(s: GameState, h: int, path: tuple[str, ...]) -> tuple[int, int | None]:
        stats.recursive_visits += 1
        if h == 0 or not s.remaining:
            stats.leaf_evaluations += 1
            if trace:
                events.append({"event": "leaf", "path": path, "score_a": s.score_a,
                               "score_b": s.score_b, "utility": s.utility})
            return s.utility, None
        best = -inf if s.player == "A" else inf
        choice = None
        for t in game.actions(s, order, stats):
            child = game.apply(s, t)
            value, _ = visit(child, h - 1, path + (game.labels[t],))
            if (s.player == "A" and value > best) or (s.player == "B" and value < best):
                best, choice = value, t
        if trace:
            events.append({"event": "backup", "path": path, "value": best,
                           "choice": game.labels[choice]})
        return best, choice

    value, target = visit(state, depth, ())
    stats.elapsed_ns = perf_counter_ns() - began
    return GameResult(game.labels[target] if target is not None else None, value, stats, events)


def treasure_alphabeta(game: TreasureGame, depth: int, *, state: GameState = GameState(),
                       order: str = "original", trace: bool = False) -> GameResult:
    _check(game, state, depth, order)
    stats, events = Metrics(), []
    began = perf_counter_ns()

    def visit(s: GameState, h: int, alpha: float, beta: float,
              path: tuple[str, ...]) -> tuple[int, int | None]:
        stats.recursive_visits += 1
        if h == 0 or not s.remaining:
            stats.leaf_evaluations += 1
            if trace:
                events.append({"event": "leaf", "path": path, "score_a": s.score_a,
                               "score_b": s.score_b, "utility": s.utility})
            return s.utility, None
        best = -inf if s.player == "A" else inf
        choice = None
        actions = game.actions(s, order, stats)
        for i, t in enumerate(actions):
            value, _ = visit(game.apply(s, t), h - 1, alpha, beta, path + (game.labels[t],))
            if s.player == "A":
                if value > best:
                    best, choice = value, t
                alpha = max(alpha, best)
            else:
                if value < best:
                    best, choice = value, t
                beta = min(beta, best)
            remaining = len(actions) - i - 1
            if alpha >= beta:
                if remaining:
                    stats.cutoffs += 1
                    stats.pruned_children += remaining
                    if trace:
                        events.append({"event": "cutoff", "path": path, "alpha": alpha,
                                       "beta": beta, "skipped_children": remaining})
                break
        if trace:
            events.append({"event": "backup", "path": path, "value": best,
                           "choice": game.labels[choice]})
        return best, choice

    value, target = visit(state, depth, -inf, inf, ())
    stats.elapsed_ns = perf_counter_ns() - began
    return GameResult(game.labels[target] if target is not None else None, value, stats, events)


def best_treasure(game: TreasureGame, depth: int) -> dict[str, str | int]:
    """题四标准输出：从题面初始状态、原始顺序求解，返回首步与收益差。"""
    result = treasure_alphabeta(game, depth)
    if result.best_target is None:
        raise ValueError("标准输出要求有可选首步；无行动时请使用详细结果接口")
    return {"best_target": result.best_target, "best_value": result.best_value}

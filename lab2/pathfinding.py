"""四邻域 A*；允许节点重开，跳过优先队列中过期条目。"""
from dataclasses import dataclass
from heapq import heappop, heappush
from itertools import count
from math import inf, isfinite
from time import perf_counter_ns
from typing import Callable, Sequence

from .metrics import Metrics

Coord = tuple[int, int]
Heuristic = Callable[[Coord], int | float]
Cell = str | int


@dataclass(frozen=True)
class Grid:
    costs: tuple[tuple[int | None, ...], ...]

    def __post_init__(self) -> None:
        if not self.costs or not self.costs[0]:
            raise ValueError("地图不能为空")
        width = len(self.costs[0])
        for row in self.costs:
            if len(row) != width:
                raise ValueError("地图必须为矩形")
            for x in row:
                if x is not None and (type(x) is not int or x < 0):
                    raise ValueError("进入代价必须是非负整数，障碍用 None")

    @property
    def shape(self) -> tuple[int, int]:
        return len(self.costs), len(self.costs[0])

    def check(self, p: Coord) -> None:
        rows, cols = self.shape
        if (not isinstance(p, tuple) or len(p) != 2 or
                any(type(x) is not int for x in p) or
                not (0 <= p[0] < rows and 0 <= p[1] < cols)):
            raise ValueError(f"无效坐标：{p}")

    def cost(self, p: Coord) -> int | None:
        return self.costs[p[0]][p[1]]

    def neighbors(self, p: Coord):
        # 固定右、下、左、上；完全相同配置的路径和节点数可复现。
        rows, cols = self.shape
        for dr, dc in ((0, 1), (1, 0), (0, -1), (-1, 0)):
            r, c = p[0] + dr, p[1] + dc
            if 0 <= r < rows and 0 <= c < cols and self.costs[r][c] is not None:
                yield (r, c)


def maze_grid(cells: Sequence[Sequence[str]]) -> Grid:
    allowed = {"S", "E", ".", "X"}
    if any(x not in allowed for row in cells for x in row):
        raise ValueError("迷宫只支持 S/E/./X")
    return Grid(tuple(tuple(None if x == "X" else 1 for x in row) for row in cells))


def weighted_grid(cells: Sequence[Sequence[Cell]], *, marker_cost: int = 1,
                  end_cost: int = 0) -> Grid:
    """题二 E=0；题四调用 end_cost=1，A/B/T1..T4 均为 1。"""
    def convert(x: Cell) -> int | None:
        if x == "X":
            return None
        if x == "E":
            return end_cost
        if x in ("S", "A", "B", "T1", "T2", "T3", "T4"):
            return marker_cost
        if type(x) is int:
            return x
        if isinstance(x, str) and x.isdecimal():
            return int(x)
        raise ValueError(f"不支持的地形：{x!r}")
    return Grid(tuple(tuple(convert(x) for x in row) for row in cells))


def build_heuristic(grid: Grid, goal: Coord, mode: str = "manhattan") -> Heuristic:
    """所有内置 h 可采纳且一致；预处理时间由实验单独记录。"""
    grid.check(goal)
    if mode not in {"zero", "manhattan", "relaxed"}:
        raise ValueError("启发函数必须是 zero/manhattan/relaxed")
    if mode == "zero" or grid.cost(goal) is None:
        return lambda p: 0
    rows, cols = grid.shape
    minimum = min((grid.cost((r, c)) for r in range(rows) for c in range(cols)
                   if (r, c) != goal and grid.cost((r, c)) is not None), default=0)
    goal_cost = grid.cost(goal)
    if mode == "manhattan":
        def h(p: Coord) -> int:
            d = abs(p[0] - goal[0]) + abs(p[1] - goal[1])
            # 至少 d 步，最后进入目标的代价单独计入；E=0 不会被高估。
            return 0 if d == 0 else (d - 1) * minimum + goal_cost
        return h
    # 轻量拓展：去掉障碍，原障碍格赋 minimum；原可走格代价不变。
    # 在更宽松的图上到目标的精确距离，是原图的下界。
    relaxed = tuple(tuple(minimum if x is None else x for x in row) for row in grid.costs)
    distance = {goal: 0}
    heap = [(0, goal)]
    while heap:
        d, q = heappop(heap)
        if d != distance[q]:
            continue
        # 反向边 u <- q 的代价仍是前向进入 q 的代价！
        for dr, dc in ((0, 1), (1, 0), (0, -1), (-1, 0)):
            u = (q[0] + dr, q[1] + dc)
            if 0 <= u[0] < rows and 0 <= u[1] < cols:
                nd = d + relaxed[q[0]][q[1]]
                if nd < distance.get(u, inf):
                    distance[u] = nd
                    heappush(heap, (nd, u))
    return lambda p: distance[p]


@dataclass
class PathResult:
    path: list[Coord]
    cost: int
    metrics: Metrics


def astar(grid: Grid, start: Coord, goal: Coord, *, heuristic: Heuristic | None = None,
          tie_break: str = "fifo") -> PathResult:
    grid.check(start)
    grid.check(goal)
    if tie_break not in {"fifo", "larger_g", "smaller_g"}:
        raise ValueError("同优先级策略必须为 fifo/larger_g/smaller_g")
    heuristic = heuristic if heuristic is not None else build_heuristic(grid, goal)
    stats = Metrics()
    began = perf_counter_ns()

    def finish(path: list[Coord], cost: int) -> PathResult:
        stats.elapsed_ns = perf_counter_ns() - began
        return PathResult(path, cost, stats)

    if grid.cost(start) is None or grid.cost(goal) is None:
        return finish([], -1)
    serial = count()
    heap: list[tuple[float, int, int, int, Coord]] = []

    def push(p: Coord, g: int) -> None:
        h = heuristic(p)
        if not isinstance(h, (int, float)) or not isfinite(h) or h < 0:
            raise ValueError("h 必须是有限非负数；可采纳性由调用者保证")
        secondary = -g if tie_break == "larger_g" else g if tie_break == "smaller_g" else 0
        heappush(heap, (g + h, secondary, next(serial), g, p))
        stats.generated_nodes += 1
        stats.frontier_peak = max(stats.frontier_peak, len(heap))

    best = {start: 0}
    parent: dict[Coord, Coord] = {}
    expanded: set[Coord] = set()
    push(start, 0)
    while heap:
        _, _, _, g, p = heappop(heap)
        stats.heap_pops += 1
        if g != best[p]:
            stats.stale_pops += 1
            continue
        if p == goal:
            path = [p]
            while p != start:
                p = parent[p]
                path.append(p)
            return finish(path[::-1], g)
        if p in expanded:
            stats.reopened_nodes += 1
        expanded.add(p)
        stats.expanded_nodes += 1
        for q in grid.neighbors(p):
            candidate = g + grid.cost(q)
            if candidate < best.get(q, inf):
                best[q] = candidate
                parent[q] = p
                push(q, candidate)
    return finish([], -1)


def astar_maze(start: Coord, goal: Coord, cells: Sequence[Sequence[str]], *,
               heuristic: str = "manhattan", tie_break: str = "fifo") -> list[Coord]:
    grid = maze_grid(cells)
    return astar(grid, start, goal, heuristic=build_heuristic(grid, goal, heuristic),
                 tie_break=tie_break).path


def astar_weighted(start: Coord, goal: Coord, cells: Sequence[Sequence[Cell]], *,
                   heuristic: str = "manhattan", tie_break: str = "fifo") -> int:
    grid = weighted_grid(cells)
    return astar(grid, start, goal, heuristic=build_heuristic(grid, goal, heuristic),
                 tie_break=tie_break).cost

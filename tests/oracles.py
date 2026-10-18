"""测试专用参考方法：不调用被测 A*、Minimax 或 Alpha-Beta。"""
from itertools import permutations
from math import inf

from lab2.pathfinding import Grid


def bellman_ford(grid: Grid, start: tuple[int, int], goal: tuple[int, int]) -> int:
    if grid.cost(start) is None or grid.cost(goal) is None:
        return -1
    rows, cols = grid.shape
    nodes = [(r, c) for r in range(rows) for c in range(cols) if grid.cost((r, c)) is not None]
    distances = {p: inf for p in nodes}
    distances[start] = 0
    # 不使用堆或 h；直接进行边松弛，作为小地图正确性参照。
    for _ in range(len(nodes) - 1):
        changed = False
        for p in nodes:
            for q in grid.neighbors(p):
                nd = distances[p] + grid.cost(q)
                if nd < distances[q]:
                    distances[q] = nd
                    changed = True
        if not changed:
            break
    return -1 if distances[goal] == inf else distances[goal]


def exhaustive_game(game, state, depth):
    """枚举所有行动排列，之后逐层分组回传；与递归搜索实现独立。"""
    h = min(depth, len(state.remaining))
    if h == 0:
        return state.utility, set()
    layer = {}
    for sequence in permutations(state.remaining, h):
        a, b, sa, sb, player = state.pos_a, state.pos_b, state.score_a, state.score_b, state.player
        for target in sequence:
            if player == "A":
                sa += game.values[target] - game.distances[a][target]
                a, player = target, "B"
            else:
                sb += game.values[target] - game.distances[b][target]
                b, player = target, "A"
        layer[sequence] = sa - sb
    first_values = None
    for level in range(h, 0, -1):
        if level == 1:
            first_values = {k[0]: v for k, v in layer.items()}
        maximizing = (state.player == "A") == ((level - 1) % 2 == 0)
        grouped = {}
        for prefix, value in layer.items():
            parent = prefix[:-1]
            grouped.setdefault(parent, []).append(value)
        layer = {p: (max(v) if maximizing else min(v)) for p, v in grouped.items()}
    value = layer[()]
    return value, {game.labels[t] for t, v in first_values.items() if v == value}

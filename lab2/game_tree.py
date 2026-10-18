"""独立递归实现 Minimax 与 Alpha-Beta；正式默认顺序为从左到右。"""
from dataclasses import dataclass, field
from math import inf, isfinite
from time import perf_counter_ns

from .metrics import Metrics


@dataclass
class TreeNode:
    value: int | float | None = None
    children: list["TreeNode"] = field(default_factory=list)


@dataclass
class TreeResult:
    value: int | float
    metrics: Metrics
    trace: list[dict]


def tree_from_dict(data: dict, maximizing: bool = True) -> TreeNode:
    data = data.get("tree_structure", data)
    if "is_maximizing_player" in data and data["is_maximizing_player"] is not maximizing:
        raise ValueError("题面要求 MAX/MIN 逐层交替")
    if data.get("children"):
        if "value" in data:
            raise ValueError("内部节点不能同时有 value")
        return TreeNode(children=[tree_from_dict(x, not maximizing) for x in data["children"]])
    node = TreeNode(value=data.get("value"))
    _leaf(node)
    return node


def _leaf(node: TreeNode) -> bool:
    if node.children:
        if node.value is not None:
            raise ValueError("内部节点不能同时有 value")
        return False
    if type(node.value) not in (int, float) or not isfinite(node.value):
        raise ValueError("叶节点必须有有限数值")
    return True


def _ordered(node: TreeNode, maximizing: bool, order: str, stats: Metrics):
    indexed = list(enumerate(node.children))
    if order == "reverse":
        return indexed[::-1]
    if order == "original":
        return indexed
    if order != "heuristic":
        raise ValueError("顺序必须为 original/reverse/heuristic")

    def estimate(child: TreeNode) -> float:
        # 仅探测两条边内的叶子均值；不是求解完整子树的完美排序。
        values = []
        frontier = [(child, 0)]
        while frontier:
            current, depth = frontier.pop()
            stats.ordering_probes += 1
            if _leaf(current):
                values.append(current.value)
            elif depth < 2:
                frontier.extend((c, depth + 1) for c in current.children)
        return sum(values) / len(values) if values else 0.0
    return sorted(indexed, key=lambda pair: estimate(pair[1]), reverse=maximizing)


def minimax(root: TreeNode, *, maximizing: bool = True, order: str = "original",
            trace: bool = False) -> TreeResult:
    if order not in {"original", "reverse", "heuristic"}:
        raise ValueError("不支持的访问顺序")
    stats, events = Metrics(), []
    began = perf_counter_ns()

    def visit(node: TreeNode, is_max: bool, path: tuple[int, ...]):
        stats.recursive_visits += 1
        if trace:
            events.append({"event": "visit", "path": path, "player": "MAX" if is_max else "MIN"})
        if _leaf(node):
            stats.leaf_evaluations += 1
            if trace:
                events.append({"event": "leaf", "path": path, "value": node.value})
            return node.value
        best = -inf if is_max else inf
        for i, child in _ordered(node, is_max, order, stats):
            value = visit(child, not is_max, path + (i,))
            best = max(best, value) if is_max else min(best, value)
        if trace:
            events.append({"event": "backup", "path": path, "value": best})
        return best

    value = visit(root, maximizing, ())
    stats.elapsed_ns = perf_counter_ns() - began
    return TreeResult(value, stats, events)


def alphabeta(root: TreeNode, *, maximizing: bool = True, order: str = "original",
              trace: bool = False) -> TreeResult:
    if order not in {"original", "reverse", "heuristic"}:
        raise ValueError("不支持的访问顺序")
    stats, events = Metrics(), []
    began = perf_counter_ns()

    def visit(node: TreeNode, is_max: bool, alpha: float, beta: float, path: tuple[int, ...]):
        stats.recursive_visits += 1
        if trace:
            events.append({"event": "visit", "path": path, "player": "MAX" if is_max else "MIN"})
        if _leaf(node):
            stats.leaf_evaluations += 1
            if trace:
                events.append({"event": "leaf", "path": path, "value": node.value})
            return node.value
        best = -inf if is_max else inf
        children = _ordered(node, is_max, order, stats)
        for position, (i, child) in enumerate(children):
            value = visit(child, not is_max, alpha, beta, path + (i,))
            if is_max:
                best = max(best, value)
                alpha = max(alpha, best)
            else:
                best = min(best, value)
                beta = min(beta, best)
            remaining = len(children) - position - 1
            if alpha >= beta:
                # 最后一个孩子后的窗口闭合没有实际剪枝，不计 cutoffs。
                if remaining:
                    stats.cutoffs += 1
                    stats.pruned_children += remaining
                    if trace:
                        events.append({"event": "cutoff", "path": path, "alpha": alpha,
                                       "beta": beta, "skipped_children": remaining})
                break
        if trace:
            events.append({"event": "backup", "path": path, "value": best})
        return best

    value = visit(root, maximizing, -inf, inf, ())
    stats.elapsed_ns = perf_counter_ns() - began
    return TreeResult(value, stats, events)


def best_score(root: TreeNode) -> int:
    """题三标准输出：根为 MAX、从左到右访问，直接返回最佳整数得分。"""
    value = alphabeta(root).value
    if isinstance(value, float) and not value.is_integer():
        raise ValueError("标准输出要求整数得分；非整数得分请使用详细结果接口")
    return int(value)

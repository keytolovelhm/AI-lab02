import json
from pathlib import Path
from random import Random

from .game_tree import TreeNode, tree_from_dict
from .pathfinding import Grid, maze_grid, weighted_grid

ROOT = Path(__file__).resolve().parents[1]


def pdf_samples() -> dict:
    return json.loads((ROOT / "data/pdf_samples.json").read_text(encoding="utf-8"))


def pdf_tree(*, shared: bool = False) -> TreeNode:
    root = tree_from_dict(pdf_samples()["tree"])
    if shared:
        # PDF 第12页构建代码重复引用同一 max2；不做记忆化，每条边分别访问。
        root.children[1].children[0] = root.children[0].children[1]
    return root


def path_scenarios(seed: int) -> list[tuple[str, Grid, tuple[int, int], tuple[int, int]]]:
    data = pdf_samples()
    output = []
    for name, builder in (("pdf_maze", maze_grid), ("pdf_weighted", weighted_grid)):
        sample = data["maze" if name == "pdf_maze" else "weighted"]
        output.append((name, builder(sample["cells"]), tuple(sample["start"]), tuple(sample["goal"])))
    output.append(("open_40", Grid(tuple((1,) * 40 for _ in range(40))), (0, 0), (39, 39)))
    rng = Random(seed)
    for name, size, weighted in (("walls_40", 40, False), ("terrain_35", 35, True)):
        cells = [[None if rng.random() < (0.13 if weighted else 0.23)
                  else rng.choice([1, 1, 1, 3, 8, 12]) if weighted else 1
                  for _ in range(size)] for _ in range(size)]
        # 确定性地保留上边界、右边界可走通道，避免只测随机无解实例。
        for i in range(size):
            # 加权场景只保证可通行，仍保留随机权值，避免人工构造廉价最短通道。
            cells[0][i] = rng.choice([1, 1, 3, 8, 12]) if weighted else 1
            cells[i][-1] = rng.choice([1, 1, 3, 8, 12]) if weighted else 1
        cells[0][0] = 1
        cells[-1][-1] = 0 if weighted else 1
        output.append((name, Grid(tuple(map(tuple, cells))), (0, 0), (size - 1, size - 1)))
    output.append(("cost_detour", Grid(((1, 15, 15, 15, 0), (1, 1, 1, 1, 1))), (0, 0), (0, 4)))
    output.append(("zero_cells", Grid(((1, 0, 0, 8, 0), (1, 1, 0, 0, 0), (2, 3, 1, 1, 0))), (0, 0), (2, 4)))
    output.append(("no_solution", Grid(tuple(tuple(None if c == 10 else 1 for c in range(20))
                                                  for _ in range(20))), (0, 0), (19, 19)))
    return output


def tree_scenarios(seed: int) -> list[tuple[str, TreeNode]]:
    rng = Random(seed)

    def make(depth: int, constant: bool = False) -> TreeNode:
        if depth == 0:
            return TreeNode(value=5 if constant else rng.randint(-30, 30))
        return TreeNode(children=[make(depth - 1, constant) for _ in range(4)])

    return [("pdf_tree", pdf_tree()), ("random_tree_a", make(5)),
            ("random_tree_b", make(5)), ("equal_leaves", make(4, True))]


def tree_to_dict(node: TreeNode) -> dict:
    return {"children": [tree_to_dict(c) for c in node.children]} if node.children else {"value": node.value}

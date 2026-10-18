# AI-Search-Lab2

《当代人工智能》实验二：使用 Python 标准库实现 A*、Minimax、Alpha-Beta，并在加权地形与竞争宝藏分配中比较正确性和搜索效率。题面样例已保存为独立 JSON，运行无需原课程 PDF。

## 四道题与标准结果

| 题目 | 实现 | 标准结果 |
|---|---|---|
| A* 迷宫 | 四方向、单位代价，返回含起终点的最短路径；无解 `[]` | 7 步，8 个坐标 |
| A* 加权地形 | 按进入目标格收费，E 的进入代价为 0；无解 `-1` | 总代价 7 |
| 博弈树 | 独立 Minimax、Alpha-Beta 递归，正式样例从左到右 | 最佳得分 5 |
| 宝藏分配 | A* 预计算六处位置的 6×6 距离矩阵，再搜索完整博弈状态 | H=2：T2 / -1；H=4：T3 / 3 |

坐标为 `(row, col)`，邻居顺序为右、下、左、上。宝藏收益为 `V-d`，终局评价为 `Score_A-Score_B`；H 计单方行动次数，经过宝藏不收集，双方互不阻挡。两个博弈算法在相同顺序下给出相同得分和最优首步；平局取当前顺序中最早的最优行动。

## 运行

Python **3.10+**，无需第三方依赖。已在 Windows / Python 3.13.7 上运行。

```bash
python run.py                    # 全部测试、标准样例、完整对照实验、SVG 图表
python run.py --tests-only       # 仅运行测试
python -m unittest discover -s tests -v
```

Windows 也可双击 `run_all.cmd`，或在 PowerShell 执行 `./run_all.ps1`。脚本自动尝试 `py -3` 与 `python`，失败返回非零退出码。

完整运行会覆盖 `results/` 中同名文件。仓库保留原实验的测量快照与报告配图；重新运行时计时会变化，固定输入的最优代价、得分、首步和节点计数应保持一致。报告的 PNG 为原快照配图，最新运行图表位于 `results/charts/`。

## 调用示例

```python
from lab2.fixtures import pdf_samples, pdf_tree
from lab2.pathfinding import astar_maze, astar_weighted
from lab2.game_tree import best_score, alphabeta
from lab2.experiments import pdf_game
from lab2.treasure import best_treasure, treasure_alphabeta

samples = pdf_samples()  # 只读取 data/pdf_samples.json
maze, terrain = samples["maze"], samples["weighted"]
path = astar_maze(tuple(maze["start"]), tuple(maze["goal"]), maze["cells"])
cost = astar_weighted(tuple(terrain["start"]), tuple(terrain["goal"]), terrain["cells"])
assert len(path) - 1 == 7 and cost == 7
assert best_score(pdf_tree()) == 5
game, paths = pdf_game()
assert best_treasure(game, 2) == {"best_target": "T2", "best_value": -1}
assert best_treasure(game, 4) == {"best_target": "T3", "best_value": 3}
detail = treasure_alphabeta(game, 4, trace=True)  # 保留详细结果、metrics、trace
tree_detail = alphabeta(pdf_tree(), trace=True)
```

## 两项轻量优化

1. **启发函数与队列平局策略**：对比 `zero`、`manhattan`、`relaxed`，以及 FIFO、较大 g 优先、较小 g 优先。加权曼哈顿下界为 `h(goal)=0`、`h(n)=(d-1)*m+c`，m 是其他可走格的最小进入代价，c 是终点进入代价，因此正确处理 E=0。`relaxed` 在移除障碍的宽松地图上反向 Dijkstra，保持可采纳性与一致性；预处理计入总耗时。
2. **Alpha-Beta 节点排序**：比较原始、逆序、简单启发式顺序。树按有限深度叶子均值估计，宝藏按当前玩家即时收益排序；排序探测和耗时均计入。排序仅影响访问顺序，不改变博弈得分。

原测量中，40×40 空地图的曼哈顿策略从 FIFO 改为较大 g 优先，扩展数 **1599→78**；宝藏 H=4 的 Alpha-Beta 启发式排序使递归访问 **47→37**、叶评 **16→12**。优化不保证更快：随机树 A 虽访问 **542→309**，排序成本使中位耗时 **484.65→1797.78 μs**。完整数据见 `results/effect_comparisons.csv` 与[实验报告](实验二实验报告.md)。

## 测试、计时与文件

38 项单元测试覆盖标准样例、无解、多条最优路径、高代价绕行、零代价环、相同叶分数、宝藏平局、状态隔离和标准输出类型；独立 Bellman-Ford 与穷举算法用于交叉验证。

`config/experiments.json` 固定种子为 **20261015**，共 **108 组配置**，每组预热 2 次、批量计时并重复 11 次，配置按 `seed+trial` 随机交错。汇总包含中位数、均值、标准差和四分位数。时间单位为 ns；预处理与搜索分别记录，总时间含启发函数和排序开销，不含输入构造、文件输出及博弈的距离预计算。

| 指标 | 口径 |
|---|---|
| `expanded_nodes` | 有效非目标节点出队并枚举邻居，包含实际重开 |
| `generated_nodes` / `heap_pops` / `stale_pops` | 入队次数（含起点）、总出队次数、过期出队次数 |
| `reopened_nodes` / `frontier_peak` | 更小 g 导致的再次展开、队列峰值 |
| `recursive_visits` / `leaf_evaluations` | 递归入口次数（含根与叶）、实际终止评价次数 |
| `cutoffs` / `pruned_children` | α≥β 触发次数、因此跳过的直接孩子数 |
| `ordering_probes` | 排序估计访问的节点或候选数 |

```text
lab2/                   核心算法、样例构造、指标、实验、SVG 制图
tests/                  单元测试与独立参考算法
data/pdf_samples.json   四题标准输入及预期结果
config/experiments.json 固定实验配置
results/                原始测量、汇总、对照、标准路径/轨迹/矩阵
results/charts/         8 张 SVG 对比图与 index.html
report_assets/          报告所用的 3 张 PNG（含剪枝示意）
实验二实验报告.md       核心分析、创新结果与局限
```

`raw_measurements.csv` 保存全部 1188 条重复测量，`summary.csv` 汇总 108 组配置，`effect_comparisons.csv` 保存 82 项对照；`scenario_inputs.json` 保存实际输入，`environment.json` 保存环境、计时方法和配置。`standard_results.json` 保留四题详细输出与搜索轨迹；距离、路径逐步代价、首步分支收益及最优行动序列各有对应结果文件。

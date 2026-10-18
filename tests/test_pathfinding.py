import random
import unittest

from lab2.fixtures import pdf_samples
from lab2.pathfinding import Grid, astar, astar_maze, astar_weighted, build_heuristic, maze_grid, weighted_grid
from tests.oracles import bellman_ford

MODES = ("zero", "manhattan", "relaxed")
TIES = ("fifo", "larger_g", "smaller_g")


class PathfindingTests(unittest.TestCase):
    def assert_path(self, grid, result, start, goal):
        if result.cost == -1:
            self.assertEqual(result.path, [])
            return
        self.assertEqual(result.path[0], start)
        self.assertEqual(result.path[-1], goal)
        self.assertEqual(sum(grid.cost(p) for p in result.path[1:]), result.cost)
        self.assertEqual(len(set(result.path)), len(result.path))
        for a, b in zip(result.path, result.path[1:]):
            self.assertIn(b, list(grid.neighbors(a)))

    def test_pdf_maze_all_strategies(self):
        s = pdf_samples()["maze"]
        grid, start, goal = maze_grid(s["cells"]), tuple(s["start"]), tuple(s["goal"])
        for h in MODES:
            for tie in TIES:
                with self.subTest(h=h, tie=tie):
                    result = astar(grid, start, goal, heuristic=build_heuristic(grid, goal, h), tie_break=tie)
                    self.assertEqual(result.cost, 7)
                    self.assertEqual(len(result.path), 8)  # 7 条边，8 个坐标。
                    self.assert_path(grid, result, start, goal)
        self.assertEqual(len(astar_maze(start, goal, s["cells"])) - 1, 7)

    def test_pdf_weighted_all_strategies(self):
        s = pdf_samples()["weighted"]
        start, goal = tuple(s["start"]), tuple(s["goal"])
        grid = weighted_grid(s["cells"])
        for h in MODES:
            for tie in TIES:
                with self.subTest(h=h, tie=tie):
                    result = astar(grid, start, goal, heuristic=build_heuristic(grid, goal, h), tie_break=tie)
                    self.assertEqual(result.cost, 7)
                    self.assert_path(grid, result, start, goal)
        self.assertEqual(astar_weighted(start, goal, s["cells"]), 7)
        self.assertEqual(sum(grid.cost(tuple(p)) for p in s["example_path"][1:]), 7)

    def test_goal_zero_and_start_not_charged(self):
        cells = [["S", "E"]]
        self.assertEqual(astar_weighted((0, 0), (0, 1), cells), 0)
        grid = weighted_grid(cells)
        self.assertEqual(build_heuristic(grid, (0, 1))((0, 0)), 0)
        self.assertEqual(astar(Grid(((50, 3),)), (0, 0), (0, 1)).cost, 3)

    def test_no_solution(self):
        cells = [["S", "X", "E"]]
        self.assertEqual(astar_maze((0, 0), (0, 2), cells), [])
        self.assertEqual(astar_weighted((0, 0), (0, 2), cells), -1)

    def test_blocked_endpoints(self):
        for start, goal in [((0, 0), (0, 1)), ((0, 1), (0, 0)), ((0, 0), (0, 0))]:
            self.assertEqual(astar(Grid(((None, 1),)), start, goal).cost, -1)

    def test_start_equals_goal(self):
        r = astar(Grid(((8,),)), (0, 0), (0, 0))
        self.assertEqual((r.path, r.cost), ([(0, 0)], 0))
        self.assertEqual(r.metrics.expanded_nodes, 0)
        self.assertEqual(r.metrics.heap_pops, 1)

    def test_multiple_optimal_paths(self):
        grid = Grid(((1, 1, 1),) * 3)
        paths = set()
        for tie in TIES:
            r = astar(grid, (0, 0), (2, 2), tie_break=tie)
            self.assertEqual(r.cost, 4)
            self.assert_path(grid, r, (0, 0), (2, 2))
            paths.add(tuple(r.path))
        self.assertGreaterEqual(len(paths), 1)

    def test_expensive_detour(self):
        grid = Grid(((1, 99, 0), (1, 1, 1)))
        for mode in MODES:
            r = astar(grid, (0, 0), (0, 2), heuristic=build_heuristic(grid, (0, 2), mode))
            self.assertEqual(r.cost, 3)
            self.assertEqual(len(r.path) - 1, 4)

    def test_zero_cost_cycle(self):
        grid = Grid(((0, 0, 0), (0, 0, 0)))
        r = astar(grid, (0, 0), (1, 2), heuristic=build_heuristic(grid, (1, 2), "relaxed"))
        self.assertEqual(r.cost, 0)
        self.assert_path(grid, r, (0, 0), (1, 2))

    def test_input_validation(self):
        for costs in ((), ((),), ((1, 1), (1,)), ((-1,),), ((True,),), ((1.5,),)):
            with self.subTest(costs=costs), self.assertRaises(ValueError):
                Grid(costs)
        grid = Grid(((1,),))
        for p in ((-1, 0), (0, 1), (1.0, 0), (True, 0)):
            with self.subTest(p=p), self.assertRaises(ValueError):
                astar(grid, p, (0, 0))
        with self.assertRaises(ValueError):
            astar(grid, (0, 0), (0, 0), tie_break="bad")
        with self.assertRaises(ValueError):
            build_heuristic(grid, (0, 0), "bad")
        with self.assertRaises(ValueError):
            astar(grid, (0, 0), (0, 0), heuristic=lambda p: float("nan"))
        with self.assertRaises(ValueError):
            weighted_grid([["unknown"]])

    def test_generated_and_expanded_count_convention(self):
        r = astar(Grid(((1, 1, 1),)), (0, 0), (0, 2))
        self.assertEqual(r.metrics.expanded_nodes, 2)
        self.assertEqual(r.metrics.generated_nodes, 3)
        self.assertEqual(r.metrics.heap_pops, 3)

    def test_admissible_inconsistent_heuristic_reopens_nodes(self):
        grid = Grid(((1, 4, 1, 1, 0), (1, 1, 1, None, None)))
        goal = (0, 4)
        h = lambda p: 4 if p == (1, 0) else 0
        self.assertEqual(bellman_ford(grid, (1, 0), goal), 4)
        result = astar(grid, (0, 0), goal, heuristic=h, tie_break="larger_g")
        self.assertEqual(result.cost, bellman_ford(grid, (0, 0), goal))
        self.assertGreater(result.metrics.reopened_nodes, 0)
        self.assert_path(grid, result, (0, 0), goal)

    def test_random_maps_against_bellman_ford(self):
        rng = random.Random(20261015)
        for case in range(45):
            cells = [[None if rng.random() < .2 else rng.choice((0, 1, 3, 8))
                      for _ in range(5)] for _ in range(4)]
            cells[0][0], cells[-1][-1] = 1, 0
            grid = Grid(tuple(map(tuple, cells)))
            expected = bellman_ford(grid, (0, 0), (3, 4))
            for mode in MODES:
                for tie in TIES:
                    with self.subTest(case=case, mode=mode, tie=tie):
                        r = astar(grid, (0, 0), (3, 4), heuristic=build_heuristic(grid, (3, 4), mode), tie_break=tie)
                        self.assertEqual(r.cost, expected)
                        self.assert_path(grid, r, (0, 0), (3, 4))

    def test_heuristic_admissibility_and_consistency(self):
        rng = random.Random(717)
        for case in range(16):
            cells = [[None if rng.random() < .15 else rng.choice((0, 1, 2, 8))
                      for _ in range(4)] for _ in range(4)]
            cells[-1][-1] = case % 3  # 不仅检查 E=0，也检查有代价的目标。
            grid, goal = Grid(tuple(map(tuple, cells))), (3, 3)
            for mode in MODES:
                h = build_heuristic(grid, goal, mode)
                self.assertEqual(h(goal), 0)
                for r in range(4):
                    for c in range(4):
                        p = (r, c)
                        if grid.cost(p) is None:
                            continue
                        exact = bellman_ford(grid, p, goal)
                        if exact != -1:
                            self.assertLessEqual(h(p), exact)
                        for q in grid.neighbors(p):
                            self.assertLessEqual(h(p), grid.cost(q) + h(q))

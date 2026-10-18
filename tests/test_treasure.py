import random
import unittest

from lab2.fixtures import pdf_samples
from lab2.pathfinding import Grid, weighted_grid
from lab2.treasure import GameState, TreasureGame, best_treasure, distance_matrix, treasure_alphabeta, treasure_minimax
from tests.oracles import bellman_ford, exhaustive_game


class TreasureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sample = pdf_samples()["treasure"]
        cls.grid = weighted_grid(cls.sample["cells"], end_cost=1)
        cls.positions = tuple(map(tuple, cls.sample["positions"]))
        cls.matrix, cls.paths = distance_matrix(cls.grid, cls.positions)
        cls.game = TreasureGame(tuple(cls.sample["labels"]), (0, 0, 9, 6, 7, 7), cls.matrix)

    def test_standard_treasure_pdf_output(self):
        for depth, expected in ((2, {"best_target": "T2", "best_value": -1}),
                                (4, {"best_target": "T3", "best_value": 3})):
            with self.subTest(depth=depth):
                before = treasure_alphabeta(self.game, depth, trace=True)
                result = best_treasure(self.game, depth)
                self.assertIs(type(result), dict)
                self.assertEqual(result, expected)
                self.assertIs(type(result["best_target"]), str)
                self.assertIs(type(result["best_value"]), int)
                after = treasure_alphabeta(self.game, depth, trace=True)
                self.assertEqual(after.metrics.counts(), before.metrics.counts())
                self.assertEqual(after.trace, before.trace)

    def test_standard_treasure_requires_a_first_action(self):
        with self.assertRaises(ValueError):
            best_treasure(self.game, 0)
        detail = treasure_alphabeta(self.game, 0)
        self.assertIsNone(detail.best_target)
        self.assertEqual(detail.best_value, 0)

    def test_all_36_distances_against_independent_oracle(self):
        for record in self.paths:
            i, j = record["from_index"], record["to_index"]
            with self.subTest(i=i, j=j):
                self.assertEqual(self.matrix[i][j], bellman_ford(self.grid, self.positions[i], self.positions[j]))
                self.assertEqual(record["cost"], sum(self.grid.cost(p) for p in record["path"][1:]))
                self.assertEqual(self.matrix[i][j], self.matrix[j][i])  # 本题端点代价都为 1。

    def test_distance_directed_enter_cost(self):
        matrix, _ = distance_matrix(Grid(((1, 7),)), ((0, 0), (0, 1)))
        self.assertEqual(matrix, ((0, 7), (1, 0)))

    def test_pdf_h2_h4_all_algorithms_orders(self):
        for depth in (2, 4):
            expected = self.sample["expected"][str(depth)]
            for solver in (treasure_minimax, treasure_alphabeta):
                for order in ("original", "reverse", "heuristic"):
                    with self.subTest(depth=depth, solver=solver.__name__, order=order):
                        r = solver(self.game, depth, order=order)
                        self.assertEqual((r.best_target, r.best_value), (expected["best_target"], expected["best_value"]))

    def test_state_update_negative_gain_and_no_side_effect(self):
        initial = GameState()
        first = self.game.apply(initial, 2)
        self.assertEqual((first.pos_a, first.pos_b, first.score_a, first.score_b, first.player), (2, 1, -1, 0, "B"))
        self.assertEqual(first.remaining, (3, 4, 5))
        second = self.game.apply(first, 4)
        self.assertEqual((second.pos_a, second.pos_b, second.score_a, second.score_b, second.player), (2, 4, -1, 3, "A"))
        self.assertEqual(initial, GameState())
        with self.assertRaises(ValueError):
            self.game.apply(first, 2)

    def test_passing_treasure_does_not_collect(self):
        # A -> T1 的最短路径会穿过 T2；只有选定的 T1 被移除。
        p = next(r["path"] for r in self.paths if r["from_index"] == 0 and r["to_index"] == 2)
        self.assertIn(self.positions[3], p)
        state = self.game.apply(GameState(), 2)
        self.assertIn(3, state.remaining)
        self.assertEqual(len(state.remaining), 3)

    def test_collected_positions_and_other_player_do_not_block(self):
        before = self.matrix
        state = self.game.apply(GameState(), 3)
        self.game.apply(state, 4)
        after, _ = distance_matrix(self.grid, self.positions)
        self.assertEqual(before, after)

    def test_horizon_and_terminal(self):
        for solver in (treasure_minimax, treasure_alphabeta):
            r = solver(self.game, 0, state=GameState(score_a=4, score_b=2))
            self.assertEqual((r.best_target, r.best_value, r.metrics.leaf_evaluations), (None, 2, 1))
            r = solver(self.game, 4, state=GameState(remaining=(), score_a=-2, score_b=3))
            self.assertEqual((r.best_target, r.best_value), (None, -5))
            self.assertEqual(solver(self.game, 10).best_value, solver(self.game, 4).best_value)

    def test_tied_treasure_choices(self):
        distances = tuple(tuple(0 if i == j else 1 for j in range(4)) for i in range(4))
        game = TreasureGame(("A", "B", "T1", "T2"), (0, 0, 3, 3), distances)
        s = GameState(remaining=(2, 3))
        for solver in (treasure_minimax, treasure_alphabeta):
            for order, target in (("original", "T1"), ("reverse", "T2"), ("heuristic", "T1")):
                r = solver(game, 2, state=s, order=order)
                self.assertEqual((r.best_value, r.best_target), (0, target))

    def test_minimax_count_and_actual_pruning(self):
        mm = treasure_minimax(self.game, 4)
        ab = treasure_alphabeta(self.game, 4)
        self.assertEqual((mm.metrics.recursive_visits, mm.metrics.leaf_evaluations), (65, 24))
        self.assertLess(ab.metrics.recursive_visits, mm.metrics.recursive_visits)
        self.assertGreater(ab.metrics.cutoffs, 0)

    def test_invalid_inputs(self):
        for solver in (treasure_minimax, treasure_alphabeta):
            for depth in (-1, 1.5, True):
                with self.assertRaises(ValueError):
                    solver(self.game, depth)
            with self.assertRaises(ValueError):
                solver(self.game, 2, state=GameState(remaining=(2, 2)))
            with self.assertRaises(ValueError):
                solver(self.game, 2, state=GameState(player="C"))
        with self.assertRaises(ValueError):
            TreasureGame(("A", "B"), (0, 0), ((0, -1), (-1, 0)))

    def test_exhaustive_permutations_pdf(self):
        for depth in (1, 2, 3, 4):
            expected, choices = exhaustive_game(self.game, GameState(), depth)
            for solver in (treasure_minimax, treasure_alphabeta):
                r = solver(self.game, depth)
                self.assertEqual(r.best_value, expected)
                self.assertIn(r.best_target, choices)

    def test_random_games_against_exhaustive_sequences(self):
        rng = random.Random(441)
        for case in range(24):
            n = 6
            distances = tuple(tuple(0 if i == j else rng.randint(1, 15) for j in range(n)) for i in range(n))
            # 对算法而言只需要确定的代价矩阵；额外允许非对称代价检验模型。
            game = TreasureGame(("A", "B", "T1", "T2", "T3", "T4"),
                                (0, 0, *(rng.randint(0, 12) for _ in range(4))), distances)
            state = GameState(score_a=case % 4, score_b=-2, player="B" if case % 2 else "A")
            for depth in (1, 2, 4):
                expected, choices = exhaustive_game(game, state, depth)
                for order in ("original", "reverse", "heuristic"):
                    for solver in (treasure_minimax, treasure_alphabeta):
                        with self.subTest(case=case, depth=depth, order=order, solver=solver.__name__):
                            r = solver(game, depth, state=state, order=order)
                            self.assertEqual(r.best_value, expected)
                            self.assertIn(r.best_target, choices)

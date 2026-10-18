import random
import unittest

from lab2.fixtures import pdf_tree, tree_scenarios
from lab2.game_tree import TreeNode, alphabeta, best_score, minimax, tree_from_dict


class TreeTests(unittest.TestCase):
    def test_standard_score_pdf_output(self):
        for shared in (False, True):
            with self.subTest(shared=shared):
                root = pdf_tree(shared=shared)
                before = alphabeta(root, trace=True)
                score = best_score(root)
                self.assertIs(type(score), int)
                self.assertEqual(score, 5)
                after = alphabeta(root, trace=True)
                self.assertEqual(after.metrics.counts(), before.metrics.counts())
                self.assertEqual(after.trace, before.trace)

    def test_standard_score_negative_and_integral_float(self):
        for value in (-7, 5.0):
            with self.subTest(value=value):
                score = best_score(TreeNode(value=value))
                self.assertIs(type(score), int)
                self.assertEqual(score, value)

    def test_standard_score_does_not_truncate_fractional_value(self):
        root = TreeNode(value=2.5)
        with self.assertRaises(ValueError):
            best_score(root)
        self.assertEqual(alphabeta(root).value, 2.5)
        self.assertEqual(minimax(root).value, 2.5)

    def test_pdf_json_and_shared_construction(self):
        for shared in (False, True):
            for solver in (minimax, alphabeta):
                with self.subTest(shared=shared, solver=solver.__name__):
                    self.assertEqual(solver(pdf_tree(shared=shared)).value, 5)

    def test_formal_left_to_right_trace(self):
        mm = minimax(pdf_tree(), trace=True)
        ab = alphabeta(pdf_tree(), trace=True)
        self.assertEqual([e["value"] for e in mm.trace if e["event"] == "leaf"], [3, 5, 6, 9, 6, 9, 1, 2])
        self.assertEqual([e["value"] for e in ab.trace if e["event"] == "leaf"], [3, 5, 6, 6, 9, 1, 2])
        self.assertEqual((mm.metrics.recursive_visits, mm.metrics.leaf_evaluations), (15, 8))
        self.assertEqual((ab.metrics.recursive_visits, ab.metrics.leaf_evaluations, ab.metrics.cutoffs), (14, 7, 1))

    def test_visit_orders_preserve_scores(self):
        for name, root in tree_scenarios(20261015):
            expected = minimax(root).value
            for order in ("original", "reverse", "heuristic"):
                with self.subTest(name=name, order=order):
                    self.assertEqual(minimax(root, order=order).value, expected)
                    result = alphabeta(root, order=order)
                    self.assertEqual(result.value, expected)
                    self.assertLessEqual(result.metrics.recursive_visits, minimax(root).metrics.recursive_visits)

    def test_single_leaf_and_min_root(self):
        for solver in (minimax, alphabeta):
            r = solver(TreeNode(value=-7))
            self.assertEqual(r.value, -7)
            self.assertEqual((r.metrics.recursive_visits, r.metrics.leaf_evaluations), (1, 1))
            root = TreeNode(children=[TreeNode(value=3), TreeNode(value=-5)])
            self.assertEqual(solver(root, maximizing=False).value, -5)

    def test_equal_leaf_scores_equality_cutoff(self):
        root = TreeNode(children=[TreeNode(children=[TreeNode(value=4), TreeNode(value=4)]) for _ in range(3)])
        r = alphabeta(root)
        self.assertEqual(r.value, 4)
        self.assertEqual(r.metrics.cutoffs, 2)
        self.assertEqual(r.metrics.leaf_evaluations, 4)

    def test_invalid_nodes_and_role_flags(self):
        for solver in (minimax, alphabeta):
            for root in (TreeNode(), TreeNode(value=float("inf")), TreeNode(value=2, children=[TreeNode(value=1)])):
                with self.assertRaises(ValueError):
                    solver(root)
            with self.assertRaises(ValueError):
                solver(TreeNode(value=1), order="bad")
        with self.assertRaises(ValueError):
            tree_from_dict({"is_maximizing_player": False, "value": 3})

    def test_random_irregular_trees(self):
        rng = random.Random(338)
        def build(d):
            return (TreeNode(value=rng.randint(-9, 9)) if d == 0 or rng.random() < .2 else
                    TreeNode(children=[build(d - 1) for _ in range(rng.randint(1, 4))]))
        def reference(n, is_max):
            if not n.children:
                return n.value
            values = [reference(c, not is_max) for c in n.children]
            return max(values) if is_max else min(values)
        for case in range(50):
            root = build(4)
            for is_max in (True, False):
                for order in ("original", "reverse", "heuristic"):
                    with self.subTest(case=case, maximizing=is_max, order=order):
                        expected = reference(root, is_max)
                        self.assertEqual(minimax(root, maximizing=is_max, order=order).value, expected)
                        self.assertEqual(alphabeta(root, maximizing=is_max, order=order).value, expected)

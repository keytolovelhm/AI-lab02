"""python run.py：单元测试、标准样例、性能对照与图表。"""
import argparse
import json
import os
import sys
import unittest
from pathlib import Path

from lab2.charts import generate_charts
from lab2.experiments import export_standard_results, run_experiments

ROOT = Path(__file__).resolve().parent


def run_tests():
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"), top_level_dir=str(ROOT))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)


def main():
    if sys.version_info < (3, 10):
        raise SystemExit("需要 Python 3.10 或更新版本")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    os.chdir(ROOT)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tests-only", action="store_true")
    args = parser.parse_args()
    run_tests()
    if args.tests_only:
        return
    export_standard_results(ROOT / "results")
    config = json.loads((ROOT / "config/experiments.json").read_text(encoding="utf-8"))
    summary = run_experiments(config, ROOT / "results")
    standard = json.loads((ROOT / "results/standard_results.json").read_text(encoding="utf-8"))
    matrix = json.loads((ROOT / "results/distance_matrix.json").read_text(encoding="utf-8"))
    generate_charts(summary, ROOT / "results/charts", standard, matrix)
    print(f"全部完成：{len(summary)} 组配置，{len(summary) * config['repeats']} 条重复测量。")
    print(f"结果目录：{ROOT / 'results'}")


if __name__ == "__main__":
    main()

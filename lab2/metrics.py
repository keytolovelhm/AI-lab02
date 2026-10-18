from dataclasses import asdict, dataclass


@dataclass
class Metrics:
    """计数口径见 README；计时不含输入构造和文件输出。"""

    expanded_nodes: int = 0
    generated_nodes: int = 0
    heap_pops: int = 0
    stale_pops: int = 0
    reopened_nodes: int = 0
    frontier_peak: int = 0
    recursive_visits: int = 0
    leaf_evaluations: int = 0
    cutoffs: int = 0
    pruned_children: int = 0
    ordering_probes: int = 0
    elapsed_ns: int = 0

    def counts(self) -> dict[str, int]:
        return {k: v for k, v in asdict(self).items() if k != "elapsed_ns"}

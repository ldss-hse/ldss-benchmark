from benchmark.task.generator.primitives.criteria_weights import generate_weights
from benchmark.task.schemas.task_scheme import ExpertDescription


def generate_expert_weights_rule():
    return {
        '0': 0.8,
        '1': 0.2,
    }


def generate_expert_weights(experts: list[ExpertDescription], all_equal: bool = False):
    raw_weights = generate_weights(len(experts), all_equal)
    weights = {}
    for expert_idx, expert in enumerate(experts):
        weights[expert.expertID] = raw_weights[expert_idx]
    return weights

import random

import numpy as np

from benchmark.task.schemas.task_scheme import CriterionDescription


def generate_criteria_weights(criteria: dict[str, list[CriterionDescription]], num_criteria_per_group: int):
    weights_per_group = {}
    for criteria_group in criteria.keys():
        random_numbers = np.random.randint(0, 10, num_criteria_per_group)
        total = np.sum(random_numbers)
        unified = [round(i, 1) for i in random_numbers / total]
        before = unified.copy()
        unified[-1] = 1 - sum(unified[:-1])

        if 0. in unified:
            unified = [1 / num_criteria_per_group for _ in range(num_criteria_per_group)]

        assert abs(1 - sum(unified)) < 0.001, f'Sum of weights should equal to 1. Before: {before}. Current: {unified}'
        weights_per_group[criteria_group] = unified
    return weights_per_group

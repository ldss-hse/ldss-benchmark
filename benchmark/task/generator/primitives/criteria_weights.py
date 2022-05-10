import numpy as np

from benchmark.task.schemas.task_scheme import CriterionDescription


def generate_weights(number_of_weights, all_equal=False):
    random_numbers = np.random.randint(0, 10, number_of_weights)
    total = np.sum(random_numbers)
    unified = [round(i, 1) for i in random_numbers / total]
    before = unified.copy()
    unified[-1] = 1 - sum(unified[:-1])

    if 0. in unified or all_equal:
        unified = [1 / number_of_weights for _ in range(number_of_weights)]

    assert abs(1 - sum(unified)) < 0.001, f'Sum of weights should equal to 1. Before: {before}. Current: {unified}'
    return unified


def generate_criteria_weights(criteria: dict[str, list[CriterionDescription]], num_criteria_per_group: int,
                              all_equal=False):
    return {
        criteria_group: generate_weights(num_criteria_per_group, all_equal) for criteria_group in criteria
    }

import random

from benchmark.task.generator.task_type import TaskType
from benchmark.task.schemas.task_scheme import CriterionDescription


def generate_criteria(num_criteria_groups: int, num_criteria_per_group: int, task_type: TaskType) \
        -> dict[str, list[CriterionDescription]]:
    res: dict[str, list[CriterionDescription]] = {}
    for criteria_group_index in range(num_criteria_groups):
        criteria_group_name = f'group_{criteria_group_index+1}'
        new_criteria: list[CriterionDescription] = []
        for criteria_index in range(num_criteria_per_group):
            name = f'Criterion {criteria_index + 1} for group {criteria_group_name}'
            is_qualitative = False
            if task_type is TaskType.NUMERIC_ONLY:
                is_qualitative = False
            if task_type is TaskType.HYBRID_FUZZY_LINGUISTIC or task_type is TaskType.HYBRID_CRISP_LINGUISTIC:
                is_qualitative = random.choice([True, False])

            new_criteria.append(CriterionDescription(criteriaID=f'{criteria_group_name}_c{criteria_index + 1}',
                                                     criteriaName=name,
                                                     qualitative=is_qualitative,
                                                     units='units',
                                                     benefit=random.choice([True, False])))
        res[criteria_group_name] = new_criteria
    return res

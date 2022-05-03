from benchmark.task.schemas.task_scheme import AbstractionLevelDescription


def _new_abstraction_level(i):
    return AbstractionLevelDescription(abstractionLevelID=f'group_{i + 1}',
                                       abstractionLevelName=f'Abstraction Level {i + 1}')


def generate_abstraction_levels(num_groups):
    return [_new_abstraction_level(i) for i in range(num_groups)]

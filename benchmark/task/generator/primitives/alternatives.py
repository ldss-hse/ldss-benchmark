import random

from benchmark.task.schemas.task_scheme import AlternativeDescription


def _new_alternative(alternative_index, num_groups):
    return AlternativeDescription(alternativeID=f'alternative_{alternative_index}',
                                  alternativeName=f'Alternative {alternative_index} name',
                                  abstractionLevelID=random.randint(0, num_groups))


def generate_alternatives(num_alternatives, num_groups):
    return [_new_alternative(i, num_groups) for i in range(num_alternatives)]
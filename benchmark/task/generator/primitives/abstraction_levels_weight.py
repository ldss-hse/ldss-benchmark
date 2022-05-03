from benchmark.task.schemas.task_scheme import AbstractionLevelDescription


def generate_abstraction_level_weights(abstraction_levels: list[AbstractionLevelDescription]):
    num_groups = len(abstraction_levels)
    all_equal_weight = 1 / num_groups
    return {
        i.abstractionLevelID: all_equal_weight for i in abstraction_levels
    }
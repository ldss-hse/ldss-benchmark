from benchmark.task.schemas.task_scheme import ExpertDescription


def _new_expert(i):
    return ExpertDescription(expertID=f'expert_{i}',
                             expertName=f'Expert {i} name',
                             competencies=['everything'])


def generate_experts(num_experts):
    return [_new_expert(i) for i in range(num_experts)]

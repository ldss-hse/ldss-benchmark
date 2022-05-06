import json
from pathlib import Path

from benchmark.task.generator.dumper import save_to_json
from benchmark.task.generator.single_task_generator import SingleTaskGenerator
from benchmark.task.generator.task_type import TaskType
from benchmark.task.schemas.task_scheme import TaskDTOScheme


def generate_tasks(experiment_tasks_path, task_type: TaskType, experiments_settings_path: Path,
                   number_of_replications_of_each_set=100):
    alternatives_range = (3, 5, 7, 9)
    criteria_range = (5, 10, 15, 20)
    num_experts = 1
    num_criteria_groups = 1

    experiment_tasks_path.mkdir(parents=True, exist_ok=True)

    task_id = 0
    for _, num_alternatives in enumerate(alternatives_range):
        for _, num_criteria in enumerate(criteria_range):
            for _ in range(number_of_replications_of_each_set):
                print(f'Generating task no. {task_id + 1}...', end=' ')
                generator = SingleTaskGenerator(num_experts=num_experts,
                                                num_alternatives=num_alternatives,
                                                num_criteria_groups=num_criteria_groups,
                                                num_criteria_per_group=num_criteria,
                                                task_type=task_type)
                res_dto: TaskDTOScheme = generator.run()

                path = experiment_tasks_path / f'task_{task_id}.json'
                save_to_json(path, res_dto)
                print('done.')
                task_id += 1

    meta_settings = dict(task_type=task_type, alternatives_range=alternatives_range, criteria_range=criteria_range,
                         num_experts=num_experts, num_criteria_groups=num_criteria_groups)
    with experiments_settings_path.open('w', encoding='utf-8') as file:
        json.dump(meta_settings, file, default=lambda x: x.name)
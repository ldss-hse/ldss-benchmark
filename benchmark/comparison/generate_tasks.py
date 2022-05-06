import json
from pathlib import Path

from benchmark.comparison.schemes.correlation_report_dto import ExperimentInfoDTO
from benchmark.task.generator.dumper import save_to_json
from benchmark.task.generator.single_task_generator import SingleTaskGenerator
from benchmark.task.generator.task_type import TaskType
from benchmark.task.schemas.task_scheme import TaskDTOScheme


def generate_tasks(experiment_tasks_path, experiments_settings_path: Path, experiment_config: ExperimentInfoDTO):
    experiment_tasks_path.mkdir(parents=True, exist_ok=True)

    task_id = 0
    for num_alternatives in experiment_config.alternatives_range:
        for num_criteria in experiment_config.criteria_range:
            for _ in range(experiment_config.num_replicas):
                print(f'Generating task no. {task_id + 1}...', end=' ')
                generator = SingleTaskGenerator(num_experts=experiment_config.num_experts,
                                                num_alternatives=num_alternatives,
                                                num_criteria_groups=experiment_config.num_criteria_groups,
                                                num_criteria_per_group=num_criteria,
                                                task_type=experiment_config.task_type)
                res_dto: TaskDTOScheme = generator.run()

                path = experiment_tasks_path / f'task_{task_id}.json'
                save_to_json(path, res_dto)
                print('done.')
                task_id += 1

    save_to_json(experiments_settings_path, experiment_config)

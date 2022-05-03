"""
Comparison follows the methodology explained in:

Blanca Ceballos, Maria T. Lamata, David A. Pelta. A comparative analysis of multi-criteria decision-making methods
doi: 10.1007/s13748-016-0093-1
"""
import json
import shutil

import numpy as np

from benchmark.constants import GENERATED_TASKS_PATH
from benchmark.methods.electre_i.core import ElectreDecisionMaker
from benchmark.methods.ml_ldm.core import MLLDMDecisionMaker
from benchmark.methods.topsis.core import TopsisDecisionMaker
from benchmark.task.generator.dumper import save_to_json
from benchmark.task.generator.single_task_generator import SingleTaskGenerator
from benchmark.task.generator.task_type import TaskType
from benchmark.task.schemas.task_scheme import TaskDTOScheme
from benchmark.task.task_model import TaskModelFactory


def generate_tasks(experiment_tasks_path):
    alternatives_range = (3, 5, 7, 9)
    criteria_range = (5, 10, 15, 20)
    number_of_replications_of_each_set = 100
    num_experts = 1
    num_criteria_groups = 1
    task_type = TaskType.NUMERIC_ONLY

    experiment_tasks_path.mkdir(parents=True, exist_ok=True)

    task_id = 0
    for alt_index, num_alternatives in enumerate(alternatives_range):
        for criteria_idx, num_criteria in enumerate(criteria_range):
            for _ in range(number_of_replications_of_each_set):
                print(f'Generating task no. {task_id+1}...', end=' ')
                generator = SingleTaskGenerator(num_experts=num_experts,
                                                num_alternatives=num_alternatives,
                                                num_criteria_groups=num_criteria_groups,
                                                num_criteria_per_group=num_criteria,
                                                task_type=task_type)
                res_dto: TaskDTOScheme = generator.run()

                path = experiment_tasks_path / f'task_{task_id}.json'
                GENERATED_TASKS_PATH.mkdir(exist_ok=True, parents=True)
                save_to_json(path, res_dto)
                print('done.')
                task_id += 1


def main():
    print('Running Experiment no. 1: only numeric assessments, single expert')
    GENERATE_NEW_DATASET = False

    experiment_root_path = GENERATED_TASKS_PATH / 'experiment_1'
    experiment_reports_path = experiment_root_path / 'reports'
    experiment_tasks_path = experiment_root_path / 'tasks'
    experiment_reports_path.mkdir(parents=True, exist_ok=True)

    if GENERATE_NEW_DATASET:
        shutil.rmtree(experiment_tasks_path)
        generate_tasks(experiment_tasks_path)

    decision_makers = {
        'ML-LDM': MLLDMDecisionMaker,
        'ELECTRE I': ElectreDecisionMaker,
        'TOPSIS': TopsisDecisionMaker
    }
    all_files = sorted(experiment_tasks_path.glob('*.json'), key=lambda f: int(f.stem.split('_')[-1]))
    num_files = len(all_files)
    for file_idx, file_path in enumerate(all_files):
        print(f'{file_idx+1}/{num_files} Processing {file_path}...', end=' ')

        task_id = int(file_path.stem.split('_')[-1])
        report_path = experiment_reports_path / f'report_{task_id}.json'

        if report_path.exists():
            print('skip.')
            continue

        task = TaskModelFactory().from_json(file_path)
        report = {}
        for decision_method_name, decision_maker_class in decision_makers.items():
            decision_maker = decision_maker_class(task)
            res = decision_maker.run()

            report[decision_method_name] = np.array(res[:, 0], dtype=int).tolist()


        with report_path.open('w', encoding='utf-8') as json_file:
            json.dump(report, json_file, indent=4)

        print('done.')


if __name__ == '__main__':
    main()

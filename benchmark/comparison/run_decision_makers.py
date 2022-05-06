import json

import numpy as np

from benchmark.comparison.methods_names import MethodsNames
from benchmark.methods.electre_i.core import ElectreDecisionMaker
from benchmark.methods.ml_ldm.core import MLLDMDecisionMaker
from benchmark.methods.topsis.core import TopsisDecisionMaker
from benchmark.task.task_model import TaskModelFactory


def run_decision_makers(experiment_tasks_path, experiment_reports_path):
    decision_makers = {
        str(MethodsNames.ML_LDM): MLLDMDecisionMaker,
        str(MethodsNames.ELECTRE_I): ElectreDecisionMaker,
        str(MethodsNames.TOPSIS): TopsisDecisionMaker
    }
    all_files = sorted(experiment_tasks_path.glob('*.json'), key=lambda f: int(f.stem.split('_')[-1]))
    num_files = len(all_files)
    for file_idx, file_path in enumerate(all_files):
        print(f'{file_idx + 1}/{num_files} Running all decision makers for task from {file_path}...', end=' ')

        task_id = int(file_path.stem.split('_')[-1])
        report_path = experiment_reports_path / f'report_{task_id}.json'

        if report_path.exists():
            print('skip.')
            continue

        task = TaskModelFactory().from_json(file_path)
        report = {
            'task_info': task.report,
        }
        is_failed = False
        reason = None
        for decision_method_name, decision_maker_class in decision_makers.items():
            decision_maker = decision_maker_class(task)
            try:
                res = decision_maker.run()
            # pylint: disable=broad-except
            except Exception:
                is_failed = True
                reason = decision_method_name
                break

            report[decision_method_name] = np.array(res[:, 0], dtype=int).tolist()
        if is_failed:
            report = {'failed': reason}

        with report_path.open('w', encoding='utf-8') as json_file:
            json.dump(report, json_file, indent=4)

        print('done.')

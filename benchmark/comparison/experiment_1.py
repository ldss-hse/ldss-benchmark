"""
Comparison follows the methodology explained in:

Blanca Ceballos, Maria T. Lamata, David A. Pelta. A comparative analysis of multi-criteria decision-making methods
doi: 10.1007/s13748-016-0093-1
"""
import enum
import json
import shutil
from typing import Optional

import numpy as np
from scipy.stats import kendalltau, spearmanr

from benchmark.comparison.correlation_report import CorrelationReport, UniqueExperimentalSetup
from benchmark.comparison.generate_tasks import generate_tasks
from benchmark.comparison.methods_names import MethodsNames
from benchmark.comparison.run_decision_makers import run_decision_makers
from benchmark.constants import GENERATED_TASKS_PATH
from benchmark.methods.electre_i.core import ElectreDecisionMaker
from benchmark.methods.ml_ldm.core import MLLDMDecisionMaker
from benchmark.methods.topsis.core import TopsisDecisionMaker
from benchmark.task.generator.dumper import save_to_json
from benchmark.task.generator.single_task_generator import SingleTaskGenerator
from benchmark.task.generator.task_type import TaskType
from benchmark.task.schemas.task_scheme import TaskDTOScheme
from benchmark.task.task_model import TaskModelFactory


def main():
    print('Running Experiment no. 1: only numeric assessments, single expert')
    GENERATE_NEW_DATASET = False
    COLLECT_DECISION_MAKING_REPORTS = False
    task_type = TaskType.NUMERIC_ONLY

    experiment_root_path = GENERATED_TASKS_PATH / 'experiment_1'
    experiment_reports_path = experiment_root_path / 'reports'
    experiment_tasks_path = experiment_root_path / 'tasks'
    experiment_reports_path.mkdir(parents=True, exist_ok=True)

    if GENERATE_NEW_DATASET:
        shutil.rmtree(experiment_tasks_path)
        generate_tasks(experiment_tasks_path, task_type)

    if COLLECT_DECISION_MAKING_REPORTS:
        run_decision_makers(experiment_tasks_path, experiment_reports_path)

    correlation_report = CorrelationReport(experiment_reports_path)
    correlation_report.build_full_raw_report()
    correlation_report.calculate_correlation()

    for unique_setup in correlation_report.unique_combinations:
        unique_setup: UniqueExperimentalSetup
        print(unique_setup)


if __name__ == '__main__':
    main()
